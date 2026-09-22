# Changelog

## V1 — Basic PDF RAG

- Single-PDF upload via Streamlit.
- PyMuPDF page-level text extraction.
- Fixed-size, per-page, overlapping word chunking.
- Local embeddings via Sentence Transformers (`all-MiniLM-L6-v2`).
- Persistent local vector storage via ChromaDB.
- Top-k retrieval and grounded generation via a local Ollama model
  (`gemma3:4b`), with an explicit "insufficient context" fallback.
- Custom dark/gold Streamlit theme (hero header, glass cards, source chips,
  status indicators) built entirely with Streamlit + CSS — no separate
  frontend framework.

**What I learned:** most of the perceived quality of a RAG answer comes from
prompt discipline ("answer only from context, say when you don't know") and
from keeping page numbers attached to every chunk all the way through
retrieval — losing that thread anywhere in the pipeline makes citations
impossible to reconstruct later.
