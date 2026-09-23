"""Local persistent ChromaDB collection for chunk storage and retrieval."""

import chromadb

from src.chunker import Chunk
from src.embeddings import embed_texts

PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "insurance_documents"


class VectorStore:
    def __init__(self, persist_dir: str = PERSIST_DIR):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

    def add_document(self, document_name: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = embed_texts([c.text for c in chunks])
        ids = [f"{document_name}::{c.chunk_index}" for c in chunks]
        metadatas = []
        for c in chunks:
            metadata = {"document_name": document_name, "page_number": c.page_number}
            # Chroma metadata values can't be None, so only add the key when detected.
            if c.printed_page_number is not None:
                metadata["printed_page_number"] = c.printed_page_number
            metadatas.append(metadata)
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=metadatas,
        )

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        query_embedding = embed_texts([question])[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
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

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)
