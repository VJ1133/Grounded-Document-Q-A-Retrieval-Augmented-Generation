"""Local persistent ChromaDB collection for chunk storage and retrieval."""

import chromadb

from src.chunker import Chunk
from src.embeddings import embed_texts

PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "insurance_documents"
DEFAULT_DOCUMENT_TYPE = "General"


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

    def query(
        self,
        question: str,
        top_k: int = 5,
        document_types: list[str] | None = None,
        document_names: list[str] | None = None,
    ) -> list[dict]:
        query_embedding = embed_texts([question])[0]

        conditions = []
        if document_types:
            conditions.append({"document_type": {"$in": document_types}})
        if document_names:
            conditions.append({"document_name": {"$in": document_names}})
        if len(conditions) > 1:
            where = {"$and": conditions}
        elif conditions:
            where = conditions[0]
        else:
            where = None

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        matches = []
        docs = result.get("documents") or [[]]
        metas = result.get("metadatas") or [[]]
        dists = result.get("distances") or [[]]
        for text, meta, distance in zip(docs[0], metas[0], dists[0]):
            page_number = meta.get("page_number")
            printed_page_number = meta.get("printed_page_number")
            matches.append(
                {
                    "text": text,
                    "document_name": meta.get("document_name"),
                    "document_type": meta.get("document_type", DEFAULT_DOCUMENT_TYPE),
                    "section": meta.get("section"),
                    "page_number": page_number,
                    "printed_page_number": printed_page_number,
                    # display_page is what citations should show to a user
                    "display_page": printed_page_number if printed_page_number else page_number,
                    "distance": distance,
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
