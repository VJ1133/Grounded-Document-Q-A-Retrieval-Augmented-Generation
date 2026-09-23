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
PDF upload ──▶ PyMuPDF (text extraction, column/table-aware)
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
Generation — Ollama (local, gemma3:4b) OR Groq API (cloud, openai/gpt-oss-120b)
  answers ONLY from retrieved context, switchable live in the UI
  │
  ▼
Answer
```

Retrieval (PDF parsing, chunking, embeddings, vector search) always runs
locally regardless of which generation provider is selected. Only the final
answer-generation call goes to Groq, and only when that provider is chosen.

## Tech stack

| Technology | Purpose |
|---|---|
| Python | Application and RAG logic |
| Streamlit | Web UI |
| Ollama | Local LLM inference (free, private, no internet required) |
| Groq API | Optional hosted LLM inference (free tier, `openai/gpt-oss-120b`) — meaningfully more accurate at precise lookups in dense/tabular context than a small local model, at the cost of sending retrieved passage text to a third party for that call |
| Sentence Transformers | Local embedding model (always runs locally, regardless of generation provider) |
| ChromaDB | Local vector database |
| PyMuPDF | PDF text extraction, with column/table-aware reading order |
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

3. **Set up at least one generation provider** (you can configure both and
   switch live in the app's sidebar):

   **Option A — Local Ollama** (free, fully private, no internet required):
   ```bash
   ollama pull gemma3:4b
   ollama serve
   ```
   (If `ollama serve` is already running as a background service, skip that
   step — the app just needs it reachable on `localhost:11434`.)

   **Option B — Groq API** (free tier, much larger model, better at precise
   lookups in dense/tabular documents — but sends retrieved passage text to
   Groq's servers for each generation call):
   1. Sign up for a free account at [console.groq.com](https://console.groq.com)
      and create an API key.
   2. Copy `.env.example` to `.env`:
      ```bash
      cp .env.example .env
      ```
   3. Open `.env` and paste your key:
      ```
      GROQ_API_KEY=your_key_here
      ```
      `.env` is git-ignored — your key is never committed.

   The app auto-detects what's available: it defaults to Groq if a key is
   configured, otherwise falls back to local Ollama. A sidebar toggle lets
   you switch between them at any time, per question.

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
- Local generation (`gemma3:4b`) is noticeably weaker than the Groq option at
  precise lookups in dense, table-heavy documents (e.g. exact values in a
  price list) — it can answer "insufficient information" even when the right
  passage was retrieved, simply from being a small model. Switch to Groq in
  the sidebar for those cases.
- Table/column-aware extraction (`src/document_loader.py`) is a heuristic,
  not a guarantee — unusual page layouts can still misattribute content.

## Privacy

PDF parsing, chunking, embeddings, and vector storage always run locally —
that part never changes. Generation depends on which provider you select:

- **Local (Ollama):** fully local, nothing leaves your machine.
- **Groq API:** the retrieved passage text (not the whole document, but
  whatever chunks were relevant to your question) is sent to Groq's servers
  for that generation call.

Do not upload confidential or copyrighted documents you don't have
permission to store or process — and if using Groq mode, don't use it for
documents you can't send to a third-party API.

## Project structure

```
insurance-rag-assistant/
├── app/
│   ├── app.py                    # Streamlit UI
│   └── assets/styles.css         # Custom UI theme
├── src/
│   ├── document_loader.py        # PDF → page text (column/table-aware PyMuPDF)
│   ├── chunker.py                # Page text → overlapping chunks
│   ├── embeddings.py             # Sentence Transformers wrapper
│   ├── vector_store.py           # ChromaDB collection wrapper
│   └── rag_pipeline.py           # Retrieval + Ollama/Groq generation
├── scripts/
│   ├── make_sample_pdf.py        # Generates a synthetic demo PDF
│   ├── debug_retrieval.py        # Inspect ranked retrieval results for a question
│   └── debug_page_extraction.py  # Flag pages that likely lost content to images
├── data/sample_documents/        # Local test PDFs (gitignored except README)
├── tests/                        # pytest unit tests
├── .env.example                  # Copy to .env and add your GROQ_API_KEY
├── requirements.txt
└── README.md
```
