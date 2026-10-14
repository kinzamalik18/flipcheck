"""Tests for flipcheck/types.py and flipcheck/interfaces.py.

Each test constructs one or more Pydantic models with valid data and checks
that defaults, field values and enum string representations match the spec.
"""

from __future__ import annotations

import pytest

from flipcheck.types import (
    GateConfig,
    ItemClass,
    ItemCounts,
    ItemResult,
    JudgeConfig,
    ProjectConfig,
    ProvidersConfig,
    RateLimitsConfig,
    Reason,
    RunSummary,
    Sample,
    SamplingConfig,
    Score,
    ScorerConfig,
    SliceSummary,
    TargetConfig,
    TestCase,
    Verdict,
    VerdictLevel,
    Version,
)

# ---------------------------------------------------------------------------
# TargetConfig
# ---------------------------------------------------------------------------


def test_target_config_llm():
    cfg = TargetConfig(
        type="llm",
        model="ollama/llama3.2:3b",
        prompt="prompts/v1.j2",
        params={"temperature": 0.7, "max_tokens": 300},
    )
    assert cfg.type == "llm"
    assert cfg.model == "ollama/llama3.2:3b"
    assert cfg.url is None
    assert cfg.callable is None
    assert cfg.mock is None


def test_target_config_http():
    cfg = TargetConfig(type="http", url="https://example.com/api")
    assert cfg.type == "http"
    assert cfg.url == "https://example.com/api"


def test_target_config_mock():
    cfg = TargetConfig(type="mock", mock={"pass_prob": {"*": 0.8}, "seed": 1})
    assert cfg.type == "mock"
    assert cfg.mock is not None
    assert cfg.mock["seed"] == 1


def test_target_config_python():
    cfg = TargetConfig(type="python", callable="myapp.targets:run")
    assert cfg.callable == "myapp.targets:run"


def test_target_config_defaults():
    cfg = TargetConfig(type="mock")
    assert cfg.params == {}
    assert cfg.system is None
    assert cfg.prompt is None


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_version():
    v = Version(
        name="baseline",
        config=TargetConfig(type="mock"),
        version_id="abc123def456",
    )
    assert v.name == "baseline"
    assert len(v.version_id) == 12
    assert isinstance(v.config, TargetConfig)


# ---------------------------------------------------------------------------
# ScorerConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scorer_type",
    ["exact", "contains", "regex", "json_schema", "json_field", "semantic", "llm_judge", "python"],
)
def test_scorer_config_all_types(scorer_type: str):
    sc = ScorerConfig(type=scorer_type)
    assert sc.type == scorer_type
    assert sc.args == {}


def test_scorer_config_with_args():
    sc = ScorerConfig(type="contains", args={"values": ["urgent"], "mode": "any"})
    assert sc.args["values"] == ["urgent"]


# ---------------------------------------------------------------------------
# TestCase
# ---------------------------------------------------------------------------


def test_test_case_minimal():
    tc = TestCase(id="t001", input={"query": "hello"})
    assert tc.id == "t001"
    assert tc.scorers == []
    assert tc.tags == []
    assert tc.critical is False
    assert tc.expected is None


def test_test_case_full():
    tc = TestCase(
        id="t002",
        input={"query": "Mujhe seene mein dard hai"},
        expected="urgent",
        scorers=[ScorerConfig(type="exact")],
        tags=["urdu", "emergency"],
        critical=True,
    )
    assert tc.critical is True
    assert "urdu" in tc.tags
    assert tc.scorers[0].type == "exact"


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


def test_sample_minimal():
    s = Sample(case_id="t001", version_id="abc123def456", sample_idx=0, output="PASS")
    assert s.case_id == "t001"
    assert s.output == "PASS"
    assert s.error is None
    assert s.cached is False
    assert s.latency_ms is None
    assert s.tokens_in is None
    assert s.tokens_out is None


def test_sample_with_error():
    s = Sample(case_id="t001", version_id="abc123def456", sample_idx=1, output=None, error="timeout")
    assert s.output is None
    assert s.error == "timeout"


def test_sample_cached():
    s = Sample(
        case_id="t001",
        version_id="abc123def456",
        sample_idx=0,
        output="hello",
        cached=True,
        latency_ms=120,
        tokens_in=10,
        tokens_out=5,
    )
    assert s.cached is True
    assert s.latency_ms == 120


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


def test_score_pass():
    sc = Score(scorer="exact", value=1.0, passed=True)
    assert sc.value == 1.0
    assert sc.passed is True
    assert sc.rationale is None


def test_score_fail_with_rationale():
    sc = Score(scorer="llm_judge", value=0.2, passed=False, rationale="Off topic.")
    assert sc.rationale == "Off topic."


# ---------------------------------------------------------------------------
# ItemCounts
# ---------------------------------------------------------------------------


def test_item_counts_minimal():
    ic = ItemCounts(case_id="t001", base_pass=4, base_n=5, cand_pass=2, cand_n=5)
    assert ic.base_first_pass is None
    assert ic.cand_first_pass is None


def test_item_counts_with_first_pass():
    ic = ItemCounts(
        case_id="t001",
        base_pass=4,
        base_n=5,
        cand_pass=2,
        cand_n=5,
        base_first_pass=True,
        cand_first_pass=False,
    )
    assert ic.base_first_pass is True
    assert ic.cand_first_pass is False


# ---------------------------------------------------------------------------
# ItemClass enum
# ---------------------------------------------------------------------------


def test_item_class_string_values():
    assert ItemClass.RELIABLE_REGRESSION == "reliable_regression"
    assert ItemClass.RELIABLE_IMPROVEMENT == "reliable_improvement"
    assert ItemClass.UNCHANGED_PASS == "unchanged_pass"
    assert ItemClass.UNCHANGED_FAIL == "unchanged_fail"
    assert ItemClass.FLAKY == "flaky"
    assert ItemClass.INCONCLUSIVE == "inconclusive"


def test_item_class_is_str():
    assert isinstance(ItemClass.RELIABLE_REGRESSION, str)


# ---------------------------------------------------------------------------
# ItemResult
# ---------------------------------------------------------------------------


def test_item_result():
    counts = ItemCounts(case_id="t001", base_pass=5, base_n=5, cand_pass=1, cand_n=5)
    result = ItemResult(
        counts=counts,
        classification=ItemClass.RELIABLE_REGRESSION,
        p_regression=0.01,
        p_improvement=None,
    )
    assert result.classification == ItemClass.RELIABLE_REGRESSION
    assert result.p_regression == 0.01
    assert result.p_improvement is None


def test_item_result_defaults():
    counts = ItemCounts(case_id="t002", base_pass=3, base_n=5, cand_pass=3, cand_n=5)
    result = ItemResult(counts=counts, classification=ItemClass.UNCHANGED_PASS)
    assert result.p_regression is None
    assert result.p_improvement is None


# ---------------------------------------------------------------------------
# SliceSummary
# ---------------------------------------------------------------------------


def test_slice_summary():
    ss = SliceSummary(
        tag="urdu",
        n_items=15,
        base_rate=0.8,
        cand_rate=0.6,
        delta=-0.2,
        regressions=3,
        improvements=0,
        low_confidence=False,
    )
    assert ss.tag == "urdu"
    assert ss.delta == pytest.approx(-0.2)
    assert ss.low_confidence is False


def test_slice_summary_low_confidence():
    ss = SliceSummary(
        tag="rare",
        n_items=5,
        base_rate=1.0,
        cand_rate=0.8,
        delta=-0.2,
        regressions=1,
        improvements=0,
        low_confidence=True,
    )
    assert ss.low_confidence is True


# ---------------------------------------------------------------------------
# RunSummary
# ---------------------------------------------------------------------------


def test_run_summary():
    ss = SliceSummary(
        tag="all",
        n_items=20,
        base_rate=0.75,
        cand_rate=0.7,
        delta=-0.05,
        regressions=1,
        improvements=0,
        low_confidence=False,
    )
    rs = RunSummary(
        n_items=20,
        base_rate=0.75,
        cand_rate=0.70,
        delta=-0.05,
        delta_ci=(-0.15, 0.05),
        mcnemar_p=0.3,
        nfr_single_shot=0.05,
        nfr_single_shot_conditional=0.07,
        reliable_nfr=0.02,
        churn_rate=0.04,
        flaky_rate=0.10,
        inconclusive_rate=0.05,
        slices=[ss],
        calls_used=200,
        est_cost_usd=0.01,
    )
    assert rs.n_items == 20
    assert rs.delta_ci == (-0.15, 0.05)
    assert rs.calls_used == 200
    assert len(rs.slices) == 1


def test_run_summary_nullable_fields():
    rs = RunSummary(
        n_items=10,
        base_rate=0.5,
        cand_rate=0.5,
        delta=0.0,
        delta_ci=(0.0, 0.0),
        mcnemar_p=None,
        nfr_single_shot=0.0,
        nfr_single_shot_conditional=0.0,
        reliable_nfr=0.0,
        churn_rate=0.0,
        flaky_rate=0.0,
        inconclusive_rate=0.0,
        slices=[],
        calls_used=0,
        est_cost_usd=None,
    )
    assert rs.mcnemar_p is None
    assert rs.est_cost_usd is None


# ---------------------------------------------------------------------------
# VerdictLevel enum
# ---------------------------------------------------------------------------


def test_verdict_level_string_values():
    assert VerdictLevel.SHIP == "ship"
    assert VerdictLevel.SHIP_WITH_CAVEATS == "ship_with_caveats"
    assert VerdictLevel.BLOCK == "block"


def test_verdict_level_is_str():
    assert isinstance(VerdictLevel.BLOCK, str)


# ---------------------------------------------------------------------------
# Reason and Verdict
# ---------------------------------------------------------------------------


def test_reason():
    r = Reason(level=VerdictLevel.BLOCK, message="Case t001 regressed reliably.")
    assert r.level == VerdictLevel.BLOCK
    assert r.case_ids == []


def test_reason_with_case_ids():
    r = Reason(
        level=VerdictLevel.SHIP_WITH_CAVEATS,
        message="Two non-critical cases regressed.",
        case_ids=["t002", "t003"],
    )
    assert len(r.case_ids) == 2


def test_verdict():
    reasons = [
        Reason(level=VerdictLevel.BLOCK, message="Critical case t001 regressed.", case_ids=["t001"]),
    ]
    v = Verdict(level=VerdictLevel.BLOCK, reasons=reasons)
    assert v.level == VerdictLevel.BLOCK
    assert len(v.reasons) == 1


def test_verdict_ship():
    v = Verdict(level=VerdictLevel.SHIP, reasons=[])
    assert v.level == VerdictLevel.SHIP
    assert v.reasons == []


# ---------------------------------------------------------------------------
# SamplingConfig
# ---------------------------------------------------------------------------


def test_sampling_config_defaults():
    sc = SamplingConfig()
    assert sc.mode == "adaptive"
    assert sc.k == 5
    assert sc.k_min == 2
    assert sc.k_max == 8
    assert sc.step == 2
    assert sc.budget_calls == 1500


def test_sampling_config_fixed():
    sc = SamplingConfig(mode="fixed", k=3)
    assert sc.mode == "fixed"
    assert sc.k == 3


# ---------------------------------------------------------------------------
# GateConfig
# ---------------------------------------------------------------------------


def test_gate_config_defaults():
    gc = GateConfig()
    assert gc.alpha == pytest.approx(0.05)
    assert gc.max_reliable_nfr == pytest.approx(0.03)
    assert gc.block_on_critical_regression is True
    assert gc.max_flaky_rate == pytest.approx(0.15)
    assert gc.max_inconclusive_rate == pytest.approx(0.10)
    assert gc.max_slice_drop == pytest.approx(0.10)


def test_gate_config_custom():
    gc = GateConfig(alpha=0.01, block_on_critical_regression=False)
    assert gc.alpha == pytest.approx(0.01)
    assert gc.block_on_critical_regression is False


# ---------------------------------------------------------------------------
# RateLimitsConfig and ProvidersConfig
# ---------------------------------------------------------------------------


def test_rate_limits_defaults():
    rl = RateLimitsConfig()
    assert rl.groq == 30
    assert rl.gemini == 10
    assert rl.ollama == 0


def test_providers_config_defaults():
    pc = ProvidersConfig()
    assert pc.concurrency == 4
    assert isinstance(pc.rate_limits, RateLimitsConfig)


# ---------------------------------------------------------------------------
# JudgeConfig
# ---------------------------------------------------------------------------


def test_judge_config_defaults():
    jc = JudgeConfig()
    assert jc.model == "gemini/gemini-2.5-flash"
    assert jc.params == {"temperature": 0}


def test_judge_config_custom():
    jc = JudgeConfig(model="groq/llama3-8b-8192", params={"temperature": 0.1})
    assert jc.model == "groq/llama3-8b-8192"


# ---------------------------------------------------------------------------
# ProjectConfig
# ---------------------------------------------------------------------------


def test_project_config_minimal():
    pc = ProjectConfig(
        project="my-llm-app",
        suite="suites/main.jsonl",
        targets={
            "baseline": TargetConfig(type="mock"),
            "candidate": TargetConfig(type="mock"),
        },
    )
    assert pc.project == "my-llm-app"
    assert pc.suite == "suites/main.jsonl"
    assert "baseline" in pc.targets
    assert "candidate" in pc.targets


def test_project_config_defaults_are_sane():
    pc = ProjectConfig(
        project="test",
        suite="suite.jsonl",
        targets={"baseline": TargetConfig(type="mock")},
    )
    assert pc.default_scorers == []
    assert pc.cache_path == ".flipcheck/flipcheck.sqlite"
    assert isinstance(pc.judge, JudgeConfig)
    assert isinstance(pc.sampling, SamplingConfig)
    assert isinstance(pc.gates, GateConfig)
    assert isinstance(pc.providers, ProvidersConfig)


def test_project_config_with_scorers():
    pc = ProjectConfig(
        project="test",
        suite="suite.jsonl",
        default_scorers=[ScorerConfig(type="exact")],
        targets={"baseline": TargetConfig(type="llm", model="ollama/llama3.2:3b")},
    )
    assert len(pc.default_scorers) == 1
    assert pc.default_scorers[0].type == "exact"


def test_project_config_judge_default_model():
    pc = ProjectConfig(
        project="x",
        suite="s.jsonl",
        targets={"b": TargetConfig(type="mock")},
    )
    assert pc.judge.model == "gemini/gemini-2.5-flash"


def test_project_config_sampling_budget():
    pc = ProjectConfig(
        project="x",
        suite="s.jsonl",
        targets={"b": TargetConfig(type="mock")},
        sampling=SamplingConfig(budget_calls=500),
    )
    assert pc.sampling.budget_calls == 500


# ---------------------------------------------------------------------------
# Interfaces import smoke test
# ---------------------------------------------------------------------------


def test_interfaces_importable():
    from flipcheck.interfaces import Provider, Scorer, Store

    assert Provider is not None
    assert Scorer is not None
    assert Store is not None
