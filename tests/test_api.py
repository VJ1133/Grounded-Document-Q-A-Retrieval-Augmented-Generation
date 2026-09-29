import io

import pytest
from fastapi.testclient import TestClient

from api import main
from src.chunker import Chunk
from src.vector_store import VectorStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = VectorStore(persist_dir=str(tmp_path / "chroma_test"))
    monkeypatch.setattr(main, "_store", store)
    return TestClient(main.app), store


def test_health_reports_document_count(client):
    c, store = client
    store.add_document(
        "policy.pdf",
        [Chunk(text="The deductible is 500 dollars.", page_number=1, chunk_index=0)],
    )

    response = c.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents_indexed"] == 1
    assert "groq_configured" in body


def test_list_documents_empty(client):
    c, _store = client
    assert c.get("/documents").json() == []


def test_ingest_and_list_document(client):
    c, _store = client
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The collision deductible is 500 dollars.")
    pdf_bytes = doc.tobytes()

    response = c.post(
        "/documents",
        files={"file": ("policy.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"document_type": "Policy"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["document_name"] == "policy.pdf"
    assert body["document_type"] == "Policy"
    assert body["chunk_count"] >= 1

    listed = c.get("/documents").json()
    assert listed == [
        {"document_name": "policy.pdf", "document_type": "Policy", "chunk_count": body["chunk_count"]}
    ]


def test_ingest_rejects_duplicate(client):
    c, store = client
    store.add_document(
        "policy.pdf", [Chunk(text="text", page_number=1, chunk_index=0)]
    )

    response = c.post(
        "/documents",
        files={"file": ("policy.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert response.status_code == 409


def test_ingest_rejects_non_pdf_bytes(client):
    c, _store = client

    response = c.post(
        "/documents",
        files={"file": ("policy.pdf", io.BytesIO(b"not a pdf"), "application/pdf")},
    )

    assert response.status_code == 422


def test_delete_document(client):
    c, store = client
    store.add_document(
        "policy.pdf", [Chunk(text="text", page_number=1, chunk_index=0)]
    )

    assert c.delete("/documents/policy.pdf").status_code == 204
    assert c.delete("/documents/policy.pdf").status_code == 404


def test_ask_with_no_documents_is_not_grounded(client):
    c, _store = client

    response = c.post("/ask", json={"question": "anything", "provider": "ollama"})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is False
    assert body["sources"] == []


def test_ask_returns_answer_and_sources(client, monkeypatch):
    c, store = client
    store.add_document(
        "policy.pdf",
        [Chunk(text="The collision deductible is 500 dollars.", page_number=1, chunk_index=0)],
    )
    monkeypatch.setattr(
        "src.rag_pipeline._PROVIDERS",
        {"ollama": lambda s, u: "The deductible is $500.\nUSED_PASSAGES: 1"},
    )

    response = c.post(
        "/ask",
        json={"question": "deductible", "provider": "ollama", "use_reranker": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The deductible is $500."
    assert body["grounded"] is True
    assert body["sources"][0]["cited"] is True
    assert body["sources"][0]["text"]


def test_ask_rejects_unknown_provider(client):
    c, _store = client

    response = c.post("/ask", json={"question": "x", "provider": "not-real"})

    assert response.status_code == 400


def test_ask_honours_history_for_followup(client, monkeypatch):
    c, store = client
    store.add_document(
        "policy.pdf",
        [Chunk(text="The collision deductible is 500 dollars.", page_number=1, chunk_index=0)],
    )
    calls = []

    def fake(system, user):
        calls.append(system)
        if system.startswith("You rewrite"):
            return "What is the collision deductible?"
        return "It is $500.\nUSED_PASSAGES: 1"

    monkeypatch.setattr("src.rag_pipeline._PROVIDERS", {"ollama": fake})

    response = c.post(
        "/ask",
        json={
            "question": "how much is it?",
            "provider": "ollama",
            "use_reranker": False,
            "history": [{"question": "Tell me about collision coverage", "answer": "ok"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["search_query"] == "What is the collision deductible?"
    assert len(calls) == 2


def test_ask_rejects_ollama_when_disabled(client, monkeypatch):
    c, _store = client
    monkeypatch.setattr(main, "ALLOW_OLLAMA", False)

    response = c.post("/ask", json={"question": "x", "provider": "ollama"})

    assert response.status_code == 400
    assert "disabled" in response.json()["detail"].lower()


def test_health_skips_ollama_check_when_disabled(client, monkeypatch):
    c, _store = client
    monkeypatch.setattr(main, "ALLOW_OLLAMA", False)

    def boom():
        raise AssertionError("should not attempt to reach Ollama when disabled")

    monkeypatch.setattr("ollama.list", boom, raising=False)

    response = c.get("/health")

    assert response.status_code == 200
    assert response.json()["ollama_reachable"] is False
