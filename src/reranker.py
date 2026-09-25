"""Second-stage reranking with a cross-encoder.

First-stage retrieval (semantic + BM25 fused via RRF) scores every chunk
against the question independently and cheaply. A cross-encoder instead reads
the question and one chunk *together* and scores how well that chunk answers
it -- more accurate, but too slow to run over the whole collection, so it is
applied only to the first stage's top candidates.
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Loaded once per process (~90 MB, downloaded on first use)."""
    return CrossEncoder(RERANKER_MODEL)


def rerank(question: str, matches: list[dict], top_k: int) -> list[dict]:
    """Reorders `matches` by cross-encoder relevance and keeps the best top_k.

    Each returned match gets a `rerank_score` (higher = more relevant). The
    sort is stable, so ties keep their first-stage order.
    """
    if len(matches) <= 1:
        return matches[:top_k]

    scores = get_reranker().predict(
        [(question, m["text"]) for m in matches], show_progress_bar=False
    )
    ranked = sorted(zip(matches, scores), key=lambda pair: pair[1], reverse=True)
    out = []
    for match, score in ranked[:top_k]:
        match["rerank_score"] = float(score)
        out.append(match)
    return out
