"""Pydantic request/response models for the FastAPI service.

Kept separate from src/ so the core pipeline (src/rag_pipeline.py,
src/vector_store.py) stays framework-agnostic -- the Streamlit app and any
future caller import from src/ directly, without pulling in FastAPI/pydantic.
"""

from pydantic import BaseModel, Field


class HistoryTurn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    provider: str = "groq"
    top_k: int = Field(default=8, ge=1, le=50)
    document_types: list[str] | None = None
    document_names: list[str] | None = None
    use_reranker: bool = True
    # Prior turns, oldest first -- mirrors answer_question(history=...).
    # A plain list-of-pairs also works over JSON; this is friendlier to read.
    history: list[HistoryTurn] | None = None


class SourceOut(BaseModel):
    text: str
    document_name: str
    document_type: str
    section: str | None = None
    page_number: int
    printed_page_number: int | None = None
    display_page: int
    distance: float
    cited: bool = False


class AskResponse(BaseModel):
    answer: str
    search_query: str
    grounded: bool
    sources: list[SourceOut]


class DocumentSummary(BaseModel):
    document_name: str
    document_type: str
    chunk_count: int


class IngestResponse(BaseModel):
    document_name: str
    document_type: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    ollama_reachable: bool
    groq_configured: bool


class ErrorResponse(BaseModel):
    detail: str
