"""Grounding test set: a curated list of questions with an expected
"should this be answerable from the indexed documents" label, run live
against whichever documents are currently indexed and the selected provider.

This is deliberately NOT a pytest unit test -- it calls a real LLM (local
Ollama or the Groq API), so it's slow/non-deterministic and would make the
fast test suite flaky. Run it by hand after ingesting the sample documents,
as a repeatable check that grounding still works the way V3 requires:
supported answers get answered, and out-of-scope questions get refused
rather than hallucinated.

Usage:
    python scripts/make_sample_pdf.py   # if you haven't already
    # then upload both sample PDFs via the app, or ingest them directly:
    python scripts/test_grounding.py --provider ollama
    python scripts/test_grounding.py --provider groq
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline import answer_question
from src.vector_store import VectorStore

# (question, should_be_answerable)
# Written against the two synthetic sample documents from make_sample_pdf.py:
# catastrophe_model_overview.pdf and sample_auto_policy_summary.pdf.
GROUNDING_TEST_SET = [
    ("What exposure characteristics does the catastrophe model use?", True),
    ("What is the collision deductible on the sample policy?", True),
    ("What are the known limitations of the synthetic catastrophe model?", True),
    ("What is the capital of France?", False),
    ("What was the stock price of Apple in 2020?", False),
    ("According to the sample policy, what is the racing exclusion?", True),
    ("What is the fifth digit of pi?", False),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["ollama", "groq"], default="ollama")
    args = parser.parse_args()

    store = VectorStore()
    if store.count() == 0:
        print("No documents indexed. Upload the sample PDFs via the app first, or run:")
        print("  python scripts/make_sample_pdf.py")
        return

    passed = 0
    for question, should_be_answerable in GROUNDING_TEST_SET:
        result = answer_question(store, question, provider=args.provider)
        ok = result.grounded == should_be_answerable
        passed += ok
        status = "PASS" if ok else "FAIL"
        expected = "answerable" if should_be_answerable else "should refuse"
        actual = "answered" if result.grounded else "refused"
        print(f"[{status}] ({expected}, got {actual}) {question}")
        if not ok:
            print(f"       answer: {result.answer[:200]}")

    print(f"\n{passed}/{len(GROUNDING_TEST_SET)} passed")


if __name__ == "__main__":
    main()
