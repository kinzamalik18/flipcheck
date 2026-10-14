"""Tests for flipcheck/scorers/ - SPEC section 7.

All scorers are tested via the public API:
  - build_scorers(case, defaults) -> list[Scorer]
  - scorer.score(case, output) -> Score

No network calls. The llm_judge scorer is tested with a fake judge function
injected via its constructor (not through build_scorers, since the spec says
the scorer must accept a custom judge function for testing).

Test-writer note: all scorer modules are stubs at the time of writing.
Tests are written from the spec, not from any implementation. They will
fail until the implementation is built.
"""

from __future__ import annotations

import importlib
import json

import pytest

from flipcheck.types import Score, ScorerConfig, TestCase

# ---------------------------------------------------------------------------
# Skip helpers: skip at module level if a scorer is not yet implemented,
# but only skip the affected test, not the whole file.
# ---------------------------------------------------------------------------


def _import_build_scorers():
    mod = importlib.import_module("flipcheck.scorers")
    fn = getattr(mod, "build_scorers", None)
    if fn is None:
        pytest.skip("flipcheck.scorers.build_scorers not implemented yet")
    return fn


def _import_exact():
    mod = importlib.import_module("flipcheck.scorers.basic")
    cls = getattr(mod, "ExactScorer", None)
    if cls is None:
        pytest.skip("ExactScorer not implemented yet")
    return cls


def _import_contains():
    mod = importlib.import_module("flipcheck.scorers.basic")
    cls = getattr(mod, "ContainsScorer", None)
    if cls is None:
        pytest.skip("ContainsScorer not implemented yet")
    return cls


def _import_regex():
    mod = importlib.import_module("flipcheck.scorers.basic")
    cls = getattr(mod, "RegexScorer", None)
    if cls is None:
        pytest.skip("RegexScorer not implemented yet")
    return cls


def _import_json_schema():
    mod = importlib.import_module("flipcheck.scorers.structured")
    cls = getattr(mod, "JsonSchemaScorer", None)
    if cls is None:
        pytest.skip("JsonSchemaScorer not implemented yet")
    return cls


def _import_json_field():
    mod = importlib.import_module("flipcheck.scorers.structured")
    cls = getattr(mod, "JsonFieldScorer", None)
    if cls is None:
        pytest.skip("JsonFieldScorer not implemented yet")
    return cls


def _import_llm_judge():
    mod = importlib.import_module("flipcheck.scorers.judge")
    cls = getattr(mod, "LLMJudgeScorer", None)
    if cls is None:
        pytest.skip("LLMJudgeScorer not implemented yet")
    return cls


def _import_python_scorer():
    mod = importlib.import_module("flipcheck.scorers.custom")
    cls = getattr(mod, "PythonScorer", None)
    if cls is None:
        pytest.skip("PythonScorer not implemented yet")
    return cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case(
    case_id: str = "t001",
    expected=None,
    scorers: list[ScorerConfig] | None = None,
) -> TestCase:
    return TestCase(
        id=case_id,
        input={"query": "test"},
        expected=expected,
        scorers=scorers or [],
    )


# ---------------------------------------------------------------------------
# exact scorer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exact_scorer_passes_case_insensitive_and_trimmed():
    """exact: 'Hello World' matches expected 'hello world' (case-insensitive, trimmed)."""
    ExactScorer = _import_exact()
    scorer = ExactScorer(args={})
    case = _make_case(expected="hello world")
    score = await scorer.score(case, "  Hello World  ")
    assert isinstance(score, Score)
    assert score.passed is True
    assert score.value == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_exact_scorer_fails_when_output_does_not_match():
    """exact: 'world' does not match expected 'hello'."""
    ExactScorer = _import_exact()
    scorer = ExactScorer(args={})
    case = _make_case(expected="hello")
    score = await scorer.score(case, "world")
    assert score.passed is False
    assert score.value == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_exact_scorer_with_field_arg_extracts_json_field():
    """exact: with field='label', extracts the 'label' key from JSON output."""
    ExactScorer = _import_exact()
    scorer = ExactScorer(args={"field": "label"})
    # expected is a dict so the scorer compares expected["label"] to parsed output["label"]
    case = _make_case(expected={"label": "urgent"})
    score = await scorer.score(case, '{"label": "urgent"}')
    assert score.passed is True


@pytest.mark.asyncio
async def test_exact_scorer_with_field_arg_fails_when_field_value_differs():
    """exact: with field='label', fails when output field value does not match expected."""
    ExactScorer = _import_exact()
    scorer = ExactScorer(args={"field": "label"})
    case = _make_case(expected={"label": "urgent"})
    score = await scorer.score(case, '{"label": "normal"}')
    assert score.passed is False


@pytest.mark.asyncio
async def test_exact_scorer_name_is_exact():
    """exact: scorer.name is 'exact'."""
    ExactScorer = _import_exact()
    scorer = ExactScorer(args={})
    assert scorer.name == "exact"


@pytest.mark.asyncio
async def test_exact_scorer_via_build_scorers():
    """exact: accessible via build_scorers with ScorerConfig(type='exact')."""
    build_scorers = _import_build_scorers()
    case = _make_case(
        expected="hello world",
        scorers=[ScorerConfig(type="exact")],
    )
    scorers = build_scorers(case, [])
    assert len(scorers) == 1
    score = await scorers[0].score(case, "Hello World")
    assert score.passed is True


# ---------------------------------------------------------------------------
# contains scorer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contains_any_passes_when_one_value_present():
    """contains mode=any: output containing 'yes' passes."""
    ContainsScorer = _import_contains()
    scorer = ContainsScorer(args={"values": ["yes", "ok"], "mode": "any"})
    case = _make_case()
    score = await scorer.score(case, "yes, that is correct")
    assert score.passed is True


@pytest.mark.asyncio
async def test_contains_any_fails_when_no_value_present():
    """contains mode=any: output with neither 'yes' nor 'ok' fails."""
    ContainsScorer = _import_contains()
    scorer = ContainsScorer(args={"values": ["yes", "ok"], "mode": "any"})
    case = _make_case()
    score = await scorer.score(case, "no answer here")
    assert score.passed is False


@pytest.mark.asyncio
async def test_contains_all_fails_when_only_one_value_present():
    """contains mode=all: output with only 'yes' fails because 'ok' is missing."""
    ContainsScorer = _import_contains()
    scorer = ContainsScorer(args={"values": ["yes", "ok"], "mode": "all"})
    case = _make_case()
    score = await scorer.score(case, "yes i agree")
    assert score.passed is False


@pytest.mark.asyncio
async def test_contains_all_passes_when_both_values_present():
    """contains mode=all: output containing both 'yes' and 'ok' passes."""
    ContainsScorer = _import_contains()
    scorer = ContainsScorer(args={"values": ["yes", "ok"], "mode": "all"})
    case = _make_case()
    score = await scorer.score(case, "yes that is ok")
    assert score.passed is True


@pytest.mark.asyncio
async def test_contains_scorer_name():
    """contains: scorer.name is 'contains'."""
    ContainsScorer = _import_contains()
    scorer = ContainsScorer(args={"values": ["x"], "mode": "any"})
    assert scorer.name == "contains"


@pytest.mark.asyncio
async def test_contains_any_passes_when_both_values_present():
    """contains mode=any: passes when both values are present (superset of any)."""
    ContainsScorer = _import_contains()
    scorer = ContainsScorer(args={"values": ["yes", "ok"], "mode": "any"})
    case = _make_case()
    score = await scorer.score(case, "yes that is ok")
    assert score.passed is True


@pytest.mark.asyncio
async def test_contains_all_empty_values_passes():
    """contains mode=all with empty values list: vacuously true, passes."""
    ContainsScorer = _import_contains()
    scorer = ContainsScorer(args={"values": [], "mode": "all"})
    case = _make_case()
    score = await scorer.score(case, "anything")
    assert score.passed is True


# ---------------------------------------------------------------------------
# regex scorer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regex_passes_when_pattern_matches():
    """regex: output 'call 555-1234 please' matches pattern r'\\d{3}-\\d{4}'."""
    RegexScorer = _import_regex()
    scorer = RegexScorer(args={"pattern": r"\d{3}-\d{4}"})
    case = _make_case()
    score = await scorer.score(case, "call 555-1234 please")
    assert score.passed is True


@pytest.mark.asyncio
async def test_regex_fails_when_pattern_does_not_match():
    """regex: output 'no number here' does not match r'\\d{3}-\\d{4}'."""
    RegexScorer = _import_regex()
    scorer = RegexScorer(args={"pattern": r"\d{3}-\d{4}"})
    case = _make_case()
    score = await scorer.score(case, "no number here")
    assert score.passed is False


@pytest.mark.asyncio
async def test_regex_must_match_false_passes_when_no_match():
    """regex must_match=False: inverted, passes when output does NOT match pattern."""
    RegexScorer = _import_regex()
    scorer = RegexScorer(args={"pattern": r"\d{3}-\d{4}", "must_match": False})
    case = _make_case()
    score = await scorer.score(case, "no number here")
    assert score.passed is True


@pytest.mark.asyncio
async def test_regex_must_match_false_fails_when_pattern_matches():
    """regex must_match=False: fails when the pattern does match (inverted)."""
    RegexScorer = _import_regex()
    scorer = RegexScorer(args={"pattern": r"\d{3}-\d{4}", "must_match": False})
    case = _make_case()
    score = await scorer.score(case, "call 555-1234 please")
    assert score.passed is False


@pytest.mark.asyncio
async def test_regex_scorer_name():
    """regex: scorer.name is 'regex'."""
    RegexScorer = _import_regex()
    scorer = RegexScorer(args={"pattern": r"\d+"})
    assert scorer.name == "regex"


@pytest.mark.asyncio
async def test_regex_must_match_default_is_true():
    """regex: must_match defaults to true so absence of pattern fails."""
    RegexScorer = _import_regex()
    scorer = RegexScorer(args={"pattern": r"\d{3}"})
    case = _make_case()
    score = await scorer.score(case, "no digits")
    assert score.passed is False


# ---------------------------------------------------------------------------
# json_schema scorer
# ---------------------------------------------------------------------------

_LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
}


@pytest.mark.asyncio
async def test_json_schema_passes_valid_output():
    """json_schema: '{"label": "urgent"}' validates against the schema."""
    JsonSchemaScorer = _import_json_schema()
    scorer = JsonSchemaScorer(args={"schema": _LABEL_SCHEMA})
    case = _make_case()
    score = await scorer.score(case, '{"label": "urgent"}')
    assert score.passed is True


@pytest.mark.asyncio
async def test_json_schema_fails_wrong_type():
    """json_schema: '{"label": 123}' fails because label must be a string."""
    JsonSchemaScorer = _import_json_schema()
    scorer = JsonSchemaScorer(args={"schema": _LABEL_SCHEMA})
    case = _make_case()
    score = await scorer.score(case, '{"label": 123}')
    assert score.passed is False


@pytest.mark.asyncio
async def test_json_schema_fails_invalid_json():
    """json_schema: non-JSON output fails."""
    JsonSchemaScorer = _import_json_schema()
    scorer = JsonSchemaScorer(args={"schema": _LABEL_SCHEMA})
    case = _make_case()
    score = await scorer.score(case, "not json")
    assert score.passed is False


@pytest.mark.asyncio
async def test_json_schema_fails_missing_required_field():
    """json_schema: '{}' fails because 'label' is required."""
    JsonSchemaScorer = _import_json_schema()
    scorer = JsonSchemaScorer(args={"schema": _LABEL_SCHEMA})
    case = _make_case()
    score = await scorer.score(case, "{}")
    assert score.passed is False


@pytest.mark.asyncio
async def test_json_schema_scorer_name():
    """json_schema: scorer.name is 'json_schema'."""
    JsonSchemaScorer = _import_json_schema()
    scorer = JsonSchemaScorer(args={"schema": _LABEL_SCHEMA})
    assert scorer.name == "json_schema"


# ---------------------------------------------------------------------------
# json_field scorer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_field_passes_when_field_matches_expected():
    """json_field: '{"label": "urgent"}' matches expected 'urgent' on field 'label'."""
    JsonFieldScorer = _import_json_field()
    scorer = JsonFieldScorer(args={"field": "label"})
    case = _make_case(expected="urgent")
    score = await scorer.score(case, '{"label": "urgent"}')
    assert score.passed is True


@pytest.mark.asyncio
async def test_json_field_fails_when_field_value_differs():
    """json_field: '{"label": "normal"}' does not match expected 'urgent'."""
    JsonFieldScorer = _import_json_field()
    scorer = JsonFieldScorer(args={"field": "label"})
    case = _make_case(expected="urgent")
    score = await scorer.score(case, '{"label": "normal"}')
    assert score.passed is False


@pytest.mark.asyncio
async def test_json_field_fails_invalid_json():
    """json_field: non-JSON output fails."""
    JsonFieldScorer = _import_json_field()
    scorer = JsonFieldScorer(args={"field": "label"})
    case = _make_case(expected="urgent")
    score = await scorer.score(case, "not json")
    assert score.passed is False


@pytest.mark.asyncio
async def test_json_field_fails_missing_field():
    """json_field: fails when the field is absent from the output object."""
    JsonFieldScorer = _import_json_field()
    scorer = JsonFieldScorer(args={"field": "label"})
    case = _make_case(expected="urgent")
    score = await scorer.score(case, '{"other": "urgent"}')
    assert score.passed is False


@pytest.mark.asyncio
async def test_json_field_scorer_name():
    """json_field: scorer.name is 'json_field'."""
    JsonFieldScorer = _import_json_field()
    scorer = JsonFieldScorer(args={"field": "label"})
    assert scorer.name == "json_field"


@pytest.mark.asyncio
async def test_json_field_case_insensitive_match():
    """json_field normalize=true: field value comparison is case-insensitive and trimmed."""
    JsonFieldScorer = _import_json_field()
    scorer = JsonFieldScorer(args={"field": "label", "normalize": True})
    case = _make_case(expected="Urgent")
    score = await scorer.score(case, '{"label": "urgent"}')
    assert score.passed is True


# ---------------------------------------------------------------------------
# llm_judge scorer (fake judge function)
# ---------------------------------------------------------------------------
#
# The spec says:
#   "The scorer must accept a custom judge function for testing."
# The scorer is expected to take a judge_fn kwarg or positional arg in its
# constructor that replaces the real LLM call for testing.
# Signature of fake: async (case, output, rubric) -> str (JSON)


async def _good_judge(case: TestCase, output: str, rubric: str) -> str:
    return json.dumps({"score": 0.9, "pass": True, "reason": "Good answer"})


async def _bad_judge(case: TestCase, output: str, rubric: str) -> str:
    return json.dumps({"score": 0.2, "pass": False, "reason": "Bad answer"})


# Returns invalid JSON on first call, valid on second (one retry).
_invalid_call_count: dict[str, int] = {}


async def _invalid_then_valid_judge(case: TestCase, output: str, rubric: str) -> str:
    key = f"{id(case)}:{output}"
    _invalid_call_count[key] = _invalid_call_count.get(key, 0) + 1
    if _invalid_call_count[key] == 1:
        return "not valid json {"
    return json.dumps({"score": 0.8, "pass": True, "reason": "Recovered"})


async def _always_invalid_judge(case: TestCase, output: str, rubric: str) -> str:
    return "not valid json {"


@pytest.mark.asyncio
async def test_llm_judge_passes_when_fake_judge_returns_pass_true():
    """llm_judge: score.passed=True when fake judge returns {'pass': true}."""
    LLMJudgeScorer = _import_llm_judge()
    scorer = LLMJudgeScorer(
        args={"rubric": "Does the output answer the question?"},
        judge_fn=_good_judge,
    )
    case = _make_case()
    score = await scorer.score(case, "The capital of France is Paris.")
    assert score.passed is True
    assert score.value == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_llm_judge_fails_when_fake_judge_returns_pass_false():
    """llm_judge: score.passed=False when fake judge returns {'pass': false}."""
    LLMJudgeScorer = _import_llm_judge()
    scorer = LLMJudgeScorer(
        args={"rubric": "Does the output answer the question?"},
        judge_fn=_bad_judge,
    )
    case = _make_case()
    score = await scorer.score(case, "I do not know.")
    assert score.passed is False
    assert score.value == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_llm_judge_rationale_comes_from_judge_reason():
    """llm_judge: score.rationale is taken from the judge's 'reason' field."""
    LLMJudgeScorer = _import_llm_judge()
    scorer = LLMJudgeScorer(
        args={"rubric": "Does the output answer the question?"},
        judge_fn=_good_judge,
    )
    case = _make_case()
    score = await scorer.score(case, "Paris.")
    assert score.rationale == "Good answer"


@pytest.mark.asyncio
async def test_llm_judge_recovers_after_one_retry_on_invalid_json():
    """llm_judge: invalid JSON on first call triggers one retry; second call succeeds."""
    _invalid_call_count.clear()
    LLMJudgeScorer = _import_llm_judge()
    scorer = LLMJudgeScorer(
        args={"rubric": "Does the output answer the question?"},
        judge_fn=_invalid_then_valid_judge,
    )
    case = _make_case()
    score = await scorer.score(case, "Some answer.")
    assert score.passed is True


@pytest.mark.asyncio
async def test_llm_judge_fails_with_parse_error_after_two_invalid_responses():
    """llm_judge: after two invalid JSON responses the score fails with parse error as rationale."""
    LLMJudgeScorer = _import_llm_judge()
    scorer = LLMJudgeScorer(
        args={"rubric": "Does the output answer the question?"},
        judge_fn=_always_invalid_judge,
    )
    case = _make_case()
    score = await scorer.score(case, "Some answer.")
    assert score.passed is False
    # rationale should mention the parse error
    assert score.rationale is not None
    assert len(score.rationale) > 0


@pytest.mark.asyncio
async def test_llm_judge_scorer_name():
    """llm_judge: scorer.name is 'llm_judge'."""
    LLMJudgeScorer = _import_llm_judge()
    scorer = LLMJudgeScorer(
        args={"rubric": "Grade this."},
        judge_fn=_good_judge,
    )
    assert scorer.name == "llm_judge"


# ---------------------------------------------------------------------------
# python scorer (custom callable)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_python_scorer_callable_returning_true_gives_passed_true():
    """python: a callable returning bool True produces Score with passed=True."""
    PythonScorer = _import_python_scorer()

    def _always_pass(case: TestCase, output: str) -> bool:
        return True

    scorer = PythonScorer(fn=_always_pass)
    case = _make_case()
    score = await scorer.score(case, "anything")
    assert isinstance(score, Score)
    assert score.passed is True


@pytest.mark.asyncio
async def test_python_scorer_callable_returning_false_gives_passed_false():
    """python: a callable returning bool False produces Score with passed=False."""
    PythonScorer = _import_python_scorer()

    def _always_fail(case: TestCase, output: str) -> bool:
        return False

    scorer = PythonScorer(fn=_always_fail)
    case = _make_case()
    score = await scorer.score(case, "anything")
    assert score.passed is False


@pytest.mark.asyncio
async def test_python_scorer_callable_returning_score_is_returned_as_is():
    """python: a callable returning a Score object passes it through unchanged."""
    PythonScorer = _import_python_scorer()

    expected_score = Score(scorer="python", value=0.75, passed=True, rationale="custom")

    def _return_score(case: TestCase, output: str) -> Score:
        return expected_score

    scorer = PythonScorer(fn=_return_score)
    case = _make_case()
    score = await scorer.score(case, "anything")
    assert score.value == pytest.approx(0.75)
    assert score.passed is True
    assert score.rationale == "custom"


@pytest.mark.asyncio
async def test_python_scorer_name():
    """python: scorer.name is 'python'."""
    PythonScorer = _import_python_scorer()
    scorer = PythonScorer(fn=lambda c, o: True)
    assert scorer.name == "python"


@pytest.mark.asyncio
async def test_python_scorer_true_value_is_1_0():
    """python: bool True return maps to score.value == 1.0."""
    PythonScorer = _import_python_scorer()
    scorer = PythonScorer(fn=lambda c, o: True)
    case = _make_case()
    score = await scorer.score(case, "x")
    assert score.value == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_python_scorer_false_value_is_0_0():
    """python: bool False return maps to score.value == 0.0."""
    PythonScorer = _import_python_scorer()
    scorer = PythonScorer(fn=lambda c, o: False)
    case = _make_case()
    score = await scorer.score(case, "x")
    assert score.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_scorers factory
# ---------------------------------------------------------------------------


def test_build_scorers_empty_case_scorers_uses_defaults():
    """build_scorers: when case.scorers is empty, default_scorers are used."""
    build_scorers = _import_build_scorers()
    case = _make_case(expected="hello", scorers=[])
    defaults = [ScorerConfig(type="exact")]
    scorers = build_scorers(case, defaults)
    assert len(scorers) == 1
    assert scorers[0].name == "exact"


def test_build_scorers_case_scorers_override_defaults():
    """build_scorers: case.scorers overrides default_scorers when non-empty."""
    build_scorers = _import_build_scorers()
    case = _make_case(
        expected="hello",
        scorers=[ScorerConfig(type="regex", args={"pattern": r"\w+"})],
    )
    defaults = [ScorerConfig(type="exact")]
    scorers = build_scorers(case, defaults)
    assert len(scorers) == 1
    assert scorers[0].name == "regex"


def test_build_scorers_empty_case_and_empty_defaults_returns_empty():
    """build_scorers: both empty -> no scorers returned."""
    build_scorers = _import_build_scorers()
    case = _make_case(scorers=[])
    scorers = build_scorers(case, [])
    assert scorers == []


def test_build_scorers_multiple_defaults_all_returned():
    """build_scorers: multiple default_scorers are all returned when case has none."""
    build_scorers = _import_build_scorers()
    case = _make_case(scorers=[])
    defaults = [
        ScorerConfig(type="exact"),
        ScorerConfig(type="regex", args={"pattern": r"\w+"}),
    ]
    scorers = build_scorers(case, defaults)
    assert len(scorers) == 2


def test_build_scorers_multiple_case_scorers_all_used():
    """build_scorers: multiple per-case scorers are all returned."""
    build_scorers = _import_build_scorers()
    case = _make_case(
        scorers=[
            ScorerConfig(type="exact"),
            ScorerConfig(type="contains", args={"values": ["x"], "mode": "any"}),
        ]
    )
    scorers = build_scorers(case, [])
    assert len(scorers) == 2


# ---------------------------------------------------------------------------
# semantic scorer (skip if sentence-transformers is not installed)
# ---------------------------------------------------------------------------


def _sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _sentence_transformers_available(),
    reason="sentence-transformers not installed (install with uv sync --extra semantic)",
)
@pytest.mark.asyncio
async def test_semantic_scorer_passes_near_duplicate():
    """semantic: a near-duplicate of expected exceeds the default threshold."""
    mod = importlib.import_module("flipcheck.scorers.semantic")
    SemanticScorer = getattr(mod, "SemanticScorer", None)
    if SemanticScorer is None:
        pytest.skip("SemanticScorer not implemented yet")
    scorer = SemanticScorer(args={"threshold": 0.8})
    case = _make_case(expected="The weather is nice today.")
    score = await scorer.score(case, "Today the weather is very nice.")
    assert score.passed is True


@pytest.mark.skipif(
    not _sentence_transformers_available(),
    reason="sentence-transformers not installed (install with uv sync --extra semantic)",
)
@pytest.mark.asyncio
async def test_semantic_scorer_fails_unrelated_output():
    """semantic: an unrelated output falls below the default threshold."""
    mod = importlib.import_module("flipcheck.scorers.semantic")
    SemanticScorer = getattr(mod, "SemanticScorer", None)
    if SemanticScorer is None:
        pytest.skip("SemanticScorer not implemented yet")
    scorer = SemanticScorer(args={"threshold": 0.8})
    case = _make_case(expected="The weather is nice today.")
    score = await scorer.score(case, "How do I compile a C++ program?")
    assert score.passed is False
