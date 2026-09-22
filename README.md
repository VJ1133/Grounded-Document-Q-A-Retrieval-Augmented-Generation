# Insurance AI Knowledge Assistant

A local, portfolio-quality Retrieval-Augmented Generation (RAG) application.
Upload insurance-related PDFs, ask questions in plain English, and get answers
grounded in your own documents — with citations back to the source page.

This project is being built version by version (V1 → V5) so that each stage
demonstrates a distinct AI engineering concept, from a minimal single-document
RAG pipeline up to an evaluated, production-style service.

> **Status: V1 — Basic PDF RAG.** Single-document upload, retrieval, and
> local LLM generation. No chat history, reranking, evaluation dashboard, or
> API yet — those arrive in later versions.

## Why this project exists

Most portfolio chatbots are thin wrappers around a hosted LLM API. This
project instead demonstrates the parts that actually make RAG hard to get
right: chunking strategy, embeddings and semantic retrieval, grounding the
model in retrieved evidence, citation accuracy, and — in later versions —
measuring retrieval and answer quality rather than asserting it.

## Architecture (V1)

```
User
  │
  ▼
Streamlit UI
  │
  ▼
PDF upload ──▶ PyMuPDF (text extraction)
  │
  ▼
Chunking (fixed-size, per-page, overlapping)
  │
  ▼
Sentence Transformers (all-MiniLM-L6-v2 embeddings)
  │
  ▼
ChromaDB (local persistent vector store)
  │
  ▼
Similarity search (top-k chunks for the question)
  │
  ▼
Ollama (local LLM, gemma3:4b) — answers ONLY from retrieved context
  │
  ▼
Answer
```

## Tech stack

| Technology | Purpose |
|---|---|
| Python | Application and RAG logic |
| Streamlit | Web UI |
| Ollama | Local LLM inference |
| Sentence Transformers | Local embedding model |
| ChromaDB | Local vector database |
| PyMuPDF | PDF text extraction |
| FastAPI | Added in V5 for API serving |
| Docker | Added in V5 for packaging |

## Roadmap

| Version | Focus |
|---|---|
| **V1** | Basic single-PDF RAG |
| V2 | Multi-document RAG with metadata filtering |
| V3 | Citations and grounded-answer formatting |
| V4 | Conversational RAG with follow-up questions |
| V5 | Evaluation, hybrid retrieval/reranking, FastAPI, tests, logging, Docker |

## Setup

1. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and run Ollama**, then pull a small instruction-following model:
   ```bash
   ollama pull gemma3:4b
   ollama serve
   ```
   (If `ollama serve` is already running as a background service, skip that
   step — the app just needs it reachable on `localhost:11434`.)

4. **(Optional) Generate a synthetic sample PDF** for a quick test drive:
   ```bash
   python scripts/make_sample_pdf.py
   ```

5. **Run the app**
   ```bash
   streamlit run app/app.py
   ```

## Example questions to try

- "What exposure characteristics does the model use?"
- "What are the known limitations of this model?"
- "What should exposure data quality checks include?"
- "What is the capital of France?" — a good test that the assistant correctly
  refuses to answer from outside knowledge.

## Known limitations (V1)

- Single-document context: uploading a second PDF is indexed but questions
  aren't yet scoped/filtered per document (that's V2).
- Fixed-size chunking (words, not semantic boundaries) — a simple starting
  point, not the final word on chunk quality.
- No conversation memory — every question is independent (V4).
- No automated retrieval/answer-quality evaluation yet (V5).
- Answer quality depends heavily on the local model (`gemma3:4b` by default);
  a larger local model will generally do better at following the
  "answer only from context" instruction.

## Privacy

Everything runs locally: PDF parsing, embeddings, vector storage, and LLM
inference. No document content is sent to a third-party API. Do not upload
confidential or copyrighted documents you don't have permission to store or
process on this machine.

## Project structure

```
insurance-rag-assistant/
├── app/
│   ├── app.py              # Streamlit UI
│   └── assets/styles.css   # Custom UI theme
├── src/
│   ├── document_loader.py  # PDF → page text (PyMuPDF)
│   ├── chunker.py          # Page text → overlapping chunks
│   ├── embeddings.py       # Sentence Transformers wrapper
│   ├── vector_store.py     # ChromaDB collection wrapper
│   └── rag_pipeline.py     # Retrieval + Ollama generation
├── scripts/
│   └── make_sample_pdf.py  # Generates a synthetic demo PDF
├── data/sample_documents/  # Local test PDFs (gitignored except README)
├── tests/                  # pytest unit tests
├── requirements.txt
└── README.md
```
