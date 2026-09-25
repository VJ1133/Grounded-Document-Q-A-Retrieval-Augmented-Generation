"""Scoring helpers for the evaluation runner (scripts/run_eval.py).

Kept free of any LLM/vector-store calls so the scoring rules themselves are
unit-testable and deterministic; the runner does the live retrieval and
generation and hands the results here.

Three independent signals per case:
  - retrieval: did a chunk from an expected page land in the top-k, and at
    what rank?
  - answer: does the answer contain every expected value and none of the
    forbidden ones (or, for out-of-scope questions, was it refused)?
  - faithfulness: is every large number in the answer actually present in
    the retrieved passages? Catches the failure `grounded` cannot -- a
    confident answer built from numbers the sources don't contain.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Numbers with 4+ digits once separators are removed. Shorter ones (years
# aside, which are filtered out below) are too common to be meaningful.
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def normalize(text: str) -> str:
    """Lower-cases and drops thousands separators and currency symbols so
    "$63,650,309" and "63650309" compare equal."""
    return re.sub(r"[,$]", "", text).lower()


def load_cases(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def retrieval_rank(sources: list[dict], expected_pages: list[int]) -> int | None:
    """1-indexed rank of the first retrieved chunk on an expected page, or
    None if none of the expected pages were retrieved."""
    for rank, source in enumerate(sources, start=1):
        if source["page_number"] in expected_pages:
            return rank
    return None


def check_values(
    answer: str, expected_values: list[str], forbidden_values: list[str]
) -> tuple[list[str], list[str]]:
    """Returns (missing_expected, present_forbidden) for the answer."""
    normalized = normalize(answer)
    missing = [v for v in expected_values if normalize(v) not in normalized]
    present = [v for v in forbidden_values if normalize(v) in normalized]
    return missing, present


def unsupported_numbers(answer: str, sources: list[dict]) -> list[str]:
    """Numbers (4+ digits, not plain years) in the answer that appear in none
    of the retrieved passages."""
    source_text = normalize(" ".join(s["text"] for s in sources))
    unsupported = []
    for match in _NUMBER_RE.findall(answer):
        digits = normalize(match).rstrip(".")
        if len(digits.replace(".", "")) < 4 or _YEAR_RE.match(digits):
            continue
        if digits not in source_text:
            unsupported.append(match)
    return unsupported


@dataclass
class CaseResult:
    case_id: str
    answerable: bool
    rank: int | None = None  # None: not retrieved (or no expected pages)
    retrieval_scored: bool = False
    answer_scored: bool = False
    answer_ok: bool | None = None
    missing_values: list[str] = field(default_factory=list)
    forbidden_present: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    grounded: bool | None = None
    search_query: str = ""
    answer: str = ""


def score_answer_case(case: dict, result) -> CaseResult:
    """Scores one full pipeline run (`result` is a RagAnswer)."""
    scored = CaseResult(
        case_id=case["id"],
        answerable=case["answerable"],
        grounded=result.grounded,
        search_query=result.search_query,
        answer=result.answer,
        answer_scored=True,
    )
    if case["answerable"]:
        scored.retrieval_scored = bool(case["expected_pages"])
        scored.rank = retrieval_rank(result.sources, case["expected_pages"])
        scored.missing_values, scored.forbidden_present = check_values(
            result.answer, case["expected_values"], case["forbidden_values"]
        )
        scored.unsupported = unsupported_numbers(result.answer, result.sources)
        scored.answer_ok = (
            result.grounded
            and not scored.missing_values
            and not scored.forbidden_present
            and not scored.unsupported
        )
    else:
        # correct behaviour for an out-of-scope question is a refusal
        scored.answer_ok = not result.grounded
    return scored


def score_retrieval_case(case: dict, sources: list[dict]) -> CaseResult:
    """Retrieval-only scoring (no LLM call)."""
    scored = CaseResult(case_id=case["id"], answerable=case["answerable"])
    if case["answerable"] and case["expected_pages"]:
        scored.retrieval_scored = True
        scored.rank = retrieval_rank(sources, case["expected_pages"])
    return scored


def summarize(results: list[CaseResult], top_k: int) -> dict:
    retrieval = [r for r in results if r.retrieval_scored]
    answers = [r for r in results if r.answer_scored]
    hits = [r for r in retrieval if r.rank is not None]
    return {
        "cases": len(results),
        "retrieval_cases": len(retrieval),
        f"hit_at_{top_k}": len(hits) / len(retrieval) if retrieval else None,
        "mrr": (sum(1 / r.rank for r in hits) / len(retrieval)) if retrieval else None,
        "answer_cases": len(answers),
        "answer_accuracy": (
            sum(1 for r in answers if r.answer_ok) / len(answers) if answers else None
        ),
        "unfaithful_answers": sum(1 for r in answers if r.unsupported),
    }
