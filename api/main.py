"""FastAPI service wrapping the RAG pipeline.

This is a second front end onto the same src/ pipeline the Streamlit app
uses (app/app.py) -- same VectorStore, same answer_question(), same
document ingestion. Nothing insurance/UI-specific lives here; it's a thin
HTTP layer so the pipeline can be called from something other than a
browser (scripts, another service, a future non-Streamlit frontend).

Run locally:
    uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.schemas import (
    AskRequest,
    AskResponse,
    DocumentSummary,
    HealthResponse,
    IngestResponse,
    SourceOut,
)
from src.chunker import chunk_pages
from src.document_loader import load_pdf_bytes
from src.rag_pipeline import answer_question
from src.vector_store import DEFAULT_DOCUMENT_TYPE, VectorStore

app = FastAPI(
    title="Insurance AI Knowledge Assistant API",
    description="Upload insurance PDFs and ask grounded, cited questions over them.",
    version="1.0.0",
)

# One VectorStore per process, same pattern as app.py's st.cache_resource --
# it opens the same on-disk Chroma collection the Streamlit app uses, so
# either front end can ingest and the other sees it immediately.
_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def _ollama_reachable() -> bool:
    try:
        import ollama

        ollama.list()
        return True
    except Exception:
        return False


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = get_store()
    return HealthResponse(
        status="ok",
        documents_indexed=len(store.document_names()),
        ollama_reachable=_ollama_reachable(),
        groq_configured=bool(os.environ.get("GROQ_API_KEY")),
    )


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    return get_store().document_summaries()


@app.post("/documents", response_model=IngestResponse, status_code=201)
async def ingest_document(
    file: UploadFile = File(...),
    document_type: str = Form(DEFAULT_DOCUMENT_TYPE),
) -> IngestResponse:
    store = get_store()
    if store.has_document(file.filename):
        raise HTTPException(
            status_code=409, detail=f"'{file.filename}' is already indexed."
        )

    pdf_bytes = await file.read()
    try:
        pages = load_pdf_bytes(pdf_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {exc}") from None

    chunks = chunk_pages(pages)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    store.add_document(file.filename, chunks, document_type=document_type.strip())
    return IngestResponse(
        document_name=file.filename,
        document_type=document_type.strip(),
        chunk_count=len(chunks),
    )


@app.delete("/documents/{document_name}", status_code=204)
def delete_document(document_name: str) -> None:
    store = get_store()
    if not store.has_document(document_name):
        raise HTTPException(status_code=404, detail=f"'{document_name}' is not indexed.")
    store.delete_document(document_name)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    store = get_store()
    if request.provider not in ("ollama", "groq"):
        raise HTTPException(
            status_code=400, detail=f"Unknown provider '{request.provider}'."
        )

    history = (
        [(turn.question, turn.answer) for turn in request.history] if request.history else None
    )
    try:
        result = answer_question(
            store,
            request.question,
            top_k=request.top_k,
            provider=request.provider,
            document_types=request.document_types,
            document_names=request.document_names,
            history=history,
            use_reranker=request.use_reranker,
        )
    except RuntimeError as exc:
        # e.g. GROQ_API_KEY missing -- a config problem, not a server bug
        raise HTTPException(status_code=502, detail=str(exc)) from None

    return AskResponse(
        answer=result.answer,
        search_query=result.search_query,
        grounded=result.grounded,
        sources=[SourceOut(**s) for s in result.sources],
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
    # Last-resort handler so an unexpected failure returns JSON, not an HTML
    # traceback page, for any programmatic caller.
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})
