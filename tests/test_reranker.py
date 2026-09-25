from src import reranker
from src.rag_pipeline import retrieve


class _FakeCrossEncoder:
    def predict(self, pairs, show_progress_bar=False):
        # score = number of query words that appear in the passage
        return [sum(w in text.lower() for w in q.lower().split()) for q, text in pairs]


def test_rerank_reorders_by_score_and_keeps_top_k(monkeypatch):
    monkeypatch.setattr(reranker, "get_reranker", lambda: _FakeCrossEncoder())
    matches = [{"text": "unrelated"}, {"text": "alabama premium"}, {"text": "alabama"}]

    out = reranker.rerank("alabama premium", matches, top_k=2)

    assert [m["text"] for m in out] == ["alabama premium", "alabama"]
    assert out[0]["rerank_score"] > out[1]["rerank_score"]


def test_rerank_passes_through_single_match():
    matches = [{"text": "only"}]
    assert reranker.rerank("q", matches, top_k=5) == matches


class _FakeStore:
    def __init__(self):
        self.calls = []

    def query(self, question, top_k, document_types=None, document_names=None):
        self.calls.append(top_k)
        return [{"text": f"c{i}"} for i in range(top_k)]


def test_retrieve_pulls_wider_pool_then_reranks(monkeypatch):
    monkeypatch.setattr(
        "src.rag_pipeline.rerank", lambda q, matches, top_k: matches[::-1][:top_k]
    )
    store = _FakeStore()

    out = retrieve(store, "q", top_k=3)

    assert store.calls == [30]
    assert [m["text"] for m in out] == ["c29", "c28", "c27"]


def test_retrieve_without_reranker_uses_top_k_directly():
    store = _FakeStore()
    assert len(retrieve(store, "q", top_k=3, use_reranker=False)) == 3
    assert store.calls == [3]


def test_retrieve_falls_back_when_reranker_fails(monkeypatch):
    def boom(q, matches, top_k):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("src.rag_pipeline.rerank", boom)
    out = retrieve(_FakeStore(), "q", top_k=3)
    assert [m["text"] for m in out] == ["c0", "c1", "c2"]
