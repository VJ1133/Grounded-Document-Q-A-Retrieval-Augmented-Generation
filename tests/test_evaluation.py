from types import SimpleNamespace

from src.evaluation import (
    check_values,
    normalize,
    retrieval_rank,
    score_answer_case,
    summarize,
    unsupported_numbers,
)


def _src(page, text="x"):
    return {"page_number": page, "text": text}


def test_normalize_ignores_separators_and_currency():
    assert normalize("$63,650,309") == normalize("63650309")


def test_retrieval_rank_is_one_indexed_and_none_on_miss():
    sources = [_src(5), _src(108), _src(9)]
    assert retrieval_rank(sources, [108]) == 2
    assert retrieval_rank(sources, [1, 2]) is None


def test_check_values_flags_missing_and_forbidden():
    missing, forbidden = check_values(
        "Premiums: $63,650,309 and exposures 2,769,093",
        ["63,650,309", "65,223,793"],
        ["2,769,093"],
    )
    assert missing == ["65,223,793"]
    assert forbidden == ["2,769,093"]


def test_unsupported_numbers_catches_numbers_absent_from_sources():
    sources = [_src(1, "Alabama | 63,650,309 | 65,223,793")]
    answer = "It was $63,650,309 in 2022 and $99,999,999 in 2021."
    assert unsupported_numbers(answer, sources) == ["99,999,999"]


def test_unsupported_numbers_ignores_years_and_short_numbers():
    assert unsupported_numbers("In 2022 there were 15 cases.", [_src(1, "unrelated")]) == []


def _case(**overrides):
    case = {
        "id": "c",
        "answerable": True,
        "expected_pages": [108],
        "expected_values": ["63,650,309"],
        "forbidden_values": ["2,769,093"],
    }
    case.update(overrides)
    return case


def test_score_answer_case_passes_correct_grounded_answer():
    result = SimpleNamespace(
        answer="It was $63,650,309.",
        grounded=True,
        search_query="q",
        sources=[_src(108, "Alabama 63,650,309")],
    )
    scored = score_answer_case(_case(), result)
    assert scored.answer_ok is True
    assert scored.rank == 1


def test_score_answer_case_fails_on_wrong_measure():
    result = SimpleNamespace(
        answer="It was 2,769,093.",
        grounded=True,
        search_query="q",
        sources=[_src(108, "63,650,309 | 2,769,093")],
    )
    scored = score_answer_case(_case(), result)
    assert scored.answer_ok is False
    assert scored.forbidden_present == ["2,769,093"]
    assert scored.missing_values == ["63,650,309"]


def test_score_answer_case_out_of_scope_requires_refusal():
    case = _case(answerable=False, expected_pages=[], expected_values=[])
    refused = SimpleNamespace(answer="no", grounded=False, search_query="q", sources=[])
    answered = SimpleNamespace(answer="Paris", grounded=True, search_query="q", sources=[])
    assert score_answer_case(case, refused).answer_ok is True
    assert score_answer_case(case, answered).answer_ok is False


def test_summarize_computes_hit_rate_and_mrr():
    hit1 = score_answer_case(
        _case(),
        SimpleNamespace(
            answer="63,650,309", grounded=True, search_query="", sources=[_src(108, "63,650,309")]
        ),
    )
    miss = score_answer_case(
        _case(),
        SimpleNamespace(answer="nope", grounded=False, search_query="", sources=[_src(1)]),
    )
    summary = summarize([hit1, miss], top_k=8)
    assert summary["hit_at_8"] == 0.5
    assert summary["mrr"] == 0.5
    assert summary["answer_accuracy"] == 0.5
