"""Local persistent ChromaDB collection for chunk storage and hybrid retrieval.

Pure semantic search struggles on chunks that are structurally similar and
differ mainly in a single exact term (a state name, a product code) -- a
chunk covering ten states' short PIP notes doesn't embed as "about Utah"
even when the literal word "Utah" is sitting right there in the text. query()
therefore combines two independent rankings over the same filtered candidate
pool and fuses them with Reciprocal Rank Fusion (RRF):
  - semantic: cosine distance between the question and each chunk's embedding
  - keyword: BM25 score between the question's tokens and each chunk's text

RRF just combines rank *positions* from each method rather than trying to
compare a cosine distance and a BM25 score on the same scale (they aren't),
which makes it robust to either signal being noisy for a given query.
"""

import re

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from src.chunker import Chunk
from src.embeddings import embed_texts

PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "insurance_documents"
DEFAULT_DOCUMENT_TYPE = "General"
RRF_K = 60  # standard smoothing constant for reciprocal rank fusion

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class VectorStore:
    def __init__(self, persist_dir: str = PERSIST_DIR):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    def add_document(
        self,
        document_name: str,
        chunks: list[Chunk],
        document_type: str = DEFAULT_DOCUMENT_TYPE,
    ) -> None:
        if not chunks:
            return
        embeddings = embed_texts([c.text for c in chunks])
        ids = [f"{document_name}::{c.chunk_index}" for c in chunks]
        metadatas = []
        for c in chunks:
            metadata = {
                "document_name": document_name,
                "page_number": c.page_number,
                "document_type": document_type or DEFAULT_DOCUMENT_TYPE,
            }
            # Chroma metadata values can't be None, so only add the key when detected.
            if c.printed_page_number is not None:
                metadata["printed_page_number"] = c.printed_page_number
            if c.section:
                metadata["section"] = c.section
            metadatas.append(metadata)
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=metadatas,
        )

    def _build_where(
        self, document_types: list[str] | None, document_names: list[str] | None
    ) -> dict | None:
        conditions = []
        if document_types:
            conditions.append({"document_type": {"$in": document_types}})
        if document_names:
            conditions.append({"document_name": {"$in": document_names}})
        if len(conditions) > 1:
            return {"$and": conditions}
        return conditions[0] if conditions else None

    def query(
        self,
        question: str,
        top_k: int = 5,
        document_types: list[str] | None = None,
        document_names: list[str] | None = None,
    ) -> list[dict]:
        where = self._build_where(document_types, document_names)

        # Pull the whole filtered candidate pool once. This is what BM25
        # needs anyway (it has to score every candidate against the query),
        # so semantic distance is computed the same way here -- brute-force
        # cosine similarity against every candidate's own embedding -- rather
        # than issuing a second, separate ANN query that could return a
        # different candidate set and leave BM25-only hits without a
        # semantic distance to report.
        pool = self._collection.get(
            where=where, include=["documents", "metadatas", "embeddings"]
        )
        ids = pool.get("ids", [])
        if not ids:
            return []
        texts = pool["documents"]
        metas = pool["metadatas"]
        embeddings = np.asarray(pool["embeddings"], dtype=float)

        query_embedding = np.asarray(embed_texts([question])[0], dtype=float)
        query_unit = query_embedding / np.linalg.norm(query_embedding)
        chunk_units = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        semantic_distances = 1.0 - (chunk_units @ query_unit)
        semantic_rank = np.argsort(semantic_distances)  # ascending: closest first

        bm25 = BM25Okapi([_tokenize(t) for t in texts])
        keyword_scores = bm25.get_scores(_tokenize(question))
        keyword_rank = np.argsort(keyword_scores)[::-1]  # descending: highest first

        fused_scores = [0.0] * len(ids)
        for rank, i in enumerate(semantic_rank):
            fused_scores[i] += 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(keyword_rank):
            fused_scores[i] += 1.0 / (RRF_K + rank + 1)

        top_indices = sorted(range(len(ids)), key=lambda i: fused_scores[i], reverse=True)
        top_indices = top_indices[:top_k]

        matches = []
        for i in top_indices:
            meta = metas[i]
            page_number = meta.get("page_number")
            printed_page_number = meta.get("printed_page_number")
            matches.append(
                {
                    "text": texts[i],
                    "document_name": meta.get("document_name"),
                    "document_type": meta.get("document_type", DEFAULT_DOCUMENT_TYPE),
                    "section": meta.get("section"),
                    "page_number": page_number,
                    "printed_page_number": printed_page_number,
                    # display_page is what citations should show to a user
                    "display_page": printed_page_number if printed_page_number else page_number,
                    "distance": float(semantic_distances[i]),
                }
            )
        return matches

    def has_document(self, document_name: str) -> bool:
        existing = self._collection.get(where={"document_name": document_name}, limit=1)
        return len(existing.get("ids", [])) > 0

    def document_names(self) -> list[str]:
        all_items = self._collection.get()
        names = {m["document_name"] for m in all_items.get("metadatas", [])}
        return sorted(names)

    def document_types(self) -> list[str]:
        all_items = self._collection.get()
        types = {
            m.get("document_type", DEFAULT_DOCUMENT_TYPE) for m in all_items.get("metadatas", [])
        }
        return sorted(types)

    def document_summaries(self) -> list[dict]:
        """One row per indexed document: name, type, and chunk count."""
        all_items = self._collection.get()
        summaries: dict[str, dict] = {}
        for meta in all_items.get("metadatas", []):
            name = meta["document_name"]
            if name not in summaries:
                summaries[name] = {
                    "document_name": name,
                    "document_type": meta.get("document_type", DEFAULT_DOCUMENT_TYPE),
                    "chunk_count": 0,
                }
            summaries[name]["chunk_count"] += 1
        return sorted(summaries.values(), key=lambda s: s["document_name"])

    def delete_document(self, document_name: str) -> None:
        self._collection.delete(where={"document_name": document_name})

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)
