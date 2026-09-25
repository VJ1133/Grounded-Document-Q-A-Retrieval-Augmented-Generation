"""Evaluation runner: scores retrieval and answers against eval/cases.json.

Like scripts/test_grounding.py this hits real retrieval (and, unless
--retrieval-only, a real LLM), so it is a manual tool, not part of pytest.
Results are also written to eval/results/<timestamp>.json so runs can be
compared before/after a change (e.g. adding reranking).

Usage:
    python scripts/run_eval.py --retrieval-only          # fast, free, no LLM
    python scripts/run_eval.py --provider groq
    python scripts/run_eval.py --provider ollama --top-k 8
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.evaluation import (  # noqa: E402
    load_cases,
    score_answer_case,
    score_retrieval_case,
    summarize,
)
from src.rag_pipeline import answer_question, retrieve  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["ollama", "groq"], default="groq")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder reranker")
    parser.add_argument("--cases", default=str(ROOT / "eval" / "cases.json"))
    args = parser.parse_args()

    spec = load_cases(args.cases)
    store = VectorStore()
    if not store.has_document(spec["document"]):
        print(f"'{spec['document']}' is not indexed. Ingest it via the app first.")
        return

    results = []
    started = time.time()
    for case in spec["cases"]:
        history = [tuple(turn) for turn in case.get("history", [])] or None

        if args.retrieval_only:
            if history:
                print(f"[skip] {case['id']} (follow-up needs the LLM to rewrite the query)")
                continue
            sources = retrieve(
                store, case["question"], top_k=args.top_k, use_reranker=not args.no_rerank
            )
            scored = score_retrieval_case(case, sources)
            rank = f"rank {scored.rank}" if scored.rank else "MISS"
            if scored.retrieval_scored:
                print(f"[{'PASS' if scored.rank else 'FAIL'}] {case['id']}: {rank}")
        else:
            result = answer_question(
                store,
                case["question"],
                top_k=args.top_k,
                provider=args.provider,
                history=history,
                use_reranker=not args.no_rerank,
            )
            scored = score_answer_case(case, result)
            print(f"[{'PASS' if scored.answer_ok else 'FAIL'}] {case['id']}")
            if not scored.answer_ok:
                if scored.missing_values:
                    print(f"       missing values: {scored.missing_values}")
                if scored.forbidden_present:
                    print(f"       forbidden values present: {scored.forbidden_present}")
                if scored.unsupported:
                    print(f"       numbers not in sources: {scored.unsupported}")
                if scored.retrieval_scored and scored.rank is None:
                    print("       expected page was NOT retrieved")
                print(f"       answer: {scored.answer[:200]!r}")
        results.append(scored)

    summary = summarize(results, args.top_k)
    print("\n--- summary ---")
    for key, value in summary.items():
        print(f"{key}: {_fmt(value) if isinstance(value, float) or value is None else value}")

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "retrieval" if args.retrieval_only else args.provider
    label += "-norerank" if args.no_rerank else "-rerank"
    out_path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{label}.json"
    out_path.write_text(
        json.dumps(
            {
                "provider": None if args.retrieval_only else args.provider,
                "top_k": args.top_k,
                "reranker": not args.no_rerank,
                "seconds": round(time.time() - started, 1),
                "summary": summary,
                "results": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
