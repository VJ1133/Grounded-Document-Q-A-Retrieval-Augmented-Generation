import pytest

from src.chunker import Chunk
from src.rag_pipeline import INSUFFICIENT_CONTEXT_MESSAGE, answer_question
from src.vector_store import VectorStore


def _make_store(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma_test"))
    store.add_document(
        "policy.pdf",
        [Chunk(text="The collision deductible is 500 dollars.", page_number=1, chunk_index=0)],
    )
    return store


def test_answer_question_no_documents_is_not_grounded(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma_test"))

    result = answer_question(store, "anything", provider="ollama")

    assert result.grounded is False
    assert result.sources == []


def test_answer_question_empty_filter_is_not_grounded(tmp_path, monkeypatch):
    store = _make_store(tmp_path)

    result = answer_question(
        store, "deductible", provider="ollama", document_types=["Nonexistent Type"]
    )

    assert result.grounded is False
    assert "filter" in result.answer.lower()


def test_answer_question_marks_refusal_as_not_grounded(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    monkeypatch.setattr(
        "src.rag_pipeline._PROVIDERS",
        {"ollama": lambda system, user: INSUFFICIENT_CONTEXT_MESSAGE},
    )

    result = answer_question(store, "deductible", provider="ollama")

    assert result.grounded is False
    assert result.sources  # evidence is still surfaced even when unsupported


def test_answer_question_marks_real_answer_as_grounded(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    monkeypatch.setattr(
        "src.rag_pipeline._PROVIDERS",
        {"ollama": lambda system, user: "The collision deductible is $500."},
    )

    result = answer_question(store, "deductible", provider="ollama")

    assert result.grounded is True
    assert result.sources


def test_answer_question_rejects_unknown_provider(tmp_path):
    store = _make_store(tmp_path)

    with pytest.raises(ValueError):
        answer_question(store, "deductible", provider="not-a-real-provider")
