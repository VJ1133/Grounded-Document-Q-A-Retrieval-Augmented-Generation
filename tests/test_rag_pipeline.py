import pytest

from src.chunker import Chunk
from src.rag_pipeline import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    _extract_used_passages,
    _strip_passage_references,
    answer_question,
)
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


def test_extract_used_passages_strips_trailer_and_parses_numbers():
    raw = "The deductible is $500.\nUSED_PASSAGES: 1,3"
    text, numbers = _extract_used_passages(raw)
    assert text == "The deductible is $500."
    assert numbers == {1, 3}


def test_extract_used_passages_handles_none():
    raw = "Insufficient info.\nUSED_PASSAGES: none"
    text, numbers = _extract_used_passages(raw)
    assert text == "Insufficient info."
    assert numbers == set()


def test_extract_used_passages_degrades_gracefully_without_trailer():
    raw = "The deductible is $500."
    text, numbers = _extract_used_passages(raw)
    assert text == "The deductible is $500."
    assert numbers == set()


def test_answer_question_marks_cited_sources(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    monkeypatch.setattr(
        "src.rag_pipeline._PROVIDERS",
        {"ollama": lambda system, user: "The deductible is $500.\nUSED_PASSAGES: 1"},
    )

    result = answer_question(store, "deductible", provider="ollama")

    assert "USED_PASSAGES" not in result.answer
    assert result.sources[0]["cited"] is True


def test_followup_is_rewritten_before_retrieval(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    calls = []

    def fake_provider(system, user):
        calls.append(system)
        if system.startswith("You rewrite"):
            return "What is the collision deductible?"
        return "It is $500.\nUSED_PASSAGES: 1"

    monkeypatch.setattr("src.rag_pipeline._PROVIDERS", {"ollama": fake_provider})

    result = answer_question(
        store,
        "how much is it?",
        provider="ollama",
        history=[("Tell me about collision coverage", "It has a deductible.")],
    )

    assert result.search_query == "What is the collision deductible?"
    assert len(calls) == 2  # rewrite + generation
    assert result.grounded is True


def test_no_history_skips_rewrite(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    calls = []
    monkeypatch.setattr(
        "src.rag_pipeline._PROVIDERS",
        {"ollama": lambda s, u: calls.append(s) or "ok\nUSED_PASSAGES: 1"},
    )

    result = answer_question(store, "deductible", provider="ollama")

    assert len(calls) == 1
    assert result.search_query == "deductible"


def test_rewrite_failure_falls_back_to_original_question(tmp_path, monkeypatch):
    store = _make_store(tmp_path)

    def flaky(system, user):
        if system.startswith("You rewrite"):
            raise RuntimeError("boom")
        return "ok\nUSED_PASSAGES: 1"

    monkeypatch.setattr("src.rag_pipeline._PROVIDERS", {"ollama": flaky})

    result = answer_question(
        store, "deductible?", provider="ollama", history=[("a", "b")]
    )

    assert result.search_query == "deductible?"


def test_strip_passage_references_removes_parentheticals():
    text = "Texas is n/a in the tables on pages 79, 80 and 82 (Passages 1, 2 and 6)."
    assert _strip_passage_references(text) == (
        "Texas is n/a in the tables on pages 79, 80 and 82."
    )
    assert _strip_passage_references("The value is 5 [Passage 3].") == "The value is 5."


def test_strip_passage_references_handles_leadin_and_recapitalises():
    text = "According to Passage 7, the earned premium is 63,650,309."
    assert _strip_passage_references(text) == "The earned premium is 63,650,309."


def test_strip_passage_references_leaves_clean_text_and_page_refs_alone():
    text = "The table on page 100 lists 26.04 for Texas."
    assert _strip_passage_references(text) == text


def test_answer_question_strips_passage_markers(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    monkeypatch.setattr(
        "src.rag_pipeline._PROVIDERS",
        {"ollama": lambda s, u: "The deductible is $500 (Passage 1).\nUSED_PASSAGES: 1"},
    )

    result = answer_question(store, "deductible", provider="ollama")

    assert result.answer == "The deductible is $500."
    assert result.sources[0]["cited"] is True
