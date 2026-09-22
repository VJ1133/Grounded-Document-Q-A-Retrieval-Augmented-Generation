"""Wraps a local Sentence Transformers model for embedding text."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Loaded once per process; Streamlit's own caching wraps this too."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()
