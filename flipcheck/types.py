"""Core data models for FlipCheck.

All Pydantic v2 models used across the package live here. This file is
orchestrator-owned: subagents must not edit it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel


class TargetConfig(BaseModel):
    """Configuration for one version target (LLM, HTTP endpoint, Python callable, or mock)."""

    type: Literal["llm", "http", "python", "mock"]
    model: str | None = None
    """LiteLLM model string, used when type is llm."""
    prompt: str | None = None
    """Path to a Jinja2 prompt template file, used when type is llm."""
    system: str | None = None
    """Optional path to a system prompt file."""
    params: dict[str, Any] = {}
    """Extra model parameters such as temperature and max_tokens."""
    url: str | None = None
    """HTTP endpoint URL, used when type is http."""
    callable: str | None = None
    """Dotted import path in the form package.module:function, used when type is python."""
    mock: dict[str, Any] | None = None
    """Mock config dict, used when type is mock. See SPEC section 6."""


class Version(BaseModel):
    """A named, hashed snapshot of a target configuration."""

    name: str
    """Human label such as baseline or candidate."""
    config: TargetConfig
    version_id: str
    """First 12 characters of the sha256 of the canonical JSON of config plus prompt file contents."""


class ScorerConfig(BaseModel):
    """Configuration for one scorer attached to a test case or suite default."""

    type: Literal[
        "exact",
        "contains",
        "regex",
        "json_schema",
        "json_field",
        "semantic",
        "llm_judge",
        "python",
    ]
    args: dict[str, Any] = {}
    """Scorer-specific arguments. See SPEC section 7 for each scorer type."""


class TestCase(BaseModel):
    """One test case in the suite."""

    id: str
    input: dict[str, Any]
    """Variables for the prompt template, or the request body for HTTP targets."""
    expected: Any | None = None
    scorers: list[ScorerConfig] = []
    """Scorers for this case. Falls back to the suite default scorers when empty."""
    tags: list[str] = []
    critical: bool = False
    """When True, a reliable regression on this case can block shipping."""


class Sample(BaseModel):
    """One model call result for a specific case, version and sample index."""

    case_id: str
    version_id: str
    sample_idx: int
    output: str | None
    error: str | None = None
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cached: bool = False
    """True when the result came from the output cache and cost no model call."""


class Score(BaseModel):
    """Result of running one scorer against one sample output."""

    scorer: str
    value: float
    """Normalized score from 0.0 (worst) to 1.0 (best)."""
    passed: bool
    rationale: str | None = None
    """Optional explanation from the scorer, required for llm_judge."""


class ItemCounts(BaseModel):
    """Pass counts for one test case across both versions."""

    case_id: str
    base_pass: int
    """Number of passing samples for the baseline version."""
    base_n: int
    """Total samples collected for the baseline version."""
    cand_pass: int
    """Number of passing samples for the candidate version."""
    cand_n: int
    """Total samples collected for the candidate version."""
    base_first_pass: bool | None = None
    """Whether sample index 0 passed for the baseline. Used for single-shot NFR."""
    cand_first_pass: bool | None = None
    """Whether sample index 0 passed for the candidate. Used for single-shot NFR."""


class ItemClass(StrEnum):
    """Classification of one test case after statistical analysis."""

    RELIABLE_REGRESSION = "reliable_regression"
    RELIABLE_IMPROVEMENT = "reliable_improvement"
    UNCHANGED_PASS = "unchanged_pass"
    UNCHANGED_FAIL = "unchanged_fail"
    FLAKY = "flaky"
    INCONCLUSIVE = "inconclusive"


class ItemResult(BaseModel):
    """Full analysis result for one test case."""

    counts: ItemCounts
    classification: ItemClass
    p_regression: float | None = None
    """Benjamini-Hochberg-adjusted p-value for a regression."""
    p_improvement: float | None = None
    """Benjamini-Hochberg-adjusted p-value for an improvement."""


class SliceSummary(BaseModel):
    """Aggregate statistics for one tag slice."""

    tag: str
    n_items: int
    base_rate: float
    """Mean pass rate for the baseline across items in this slice."""
    cand_rate: float
    """Mean pass rate for the candidate across items in this slice."""
    delta: float
    """cand_rate minus base_rate."""
    regressions: int
    improvements: int
    low_confidence: bool
    """True when n_items is below 10, meaning the slice estimate is unreliable."""


class RunSummary(BaseModel):
    """Aggregate statistics for a full comparison run."""

    n_items: int
    base_rate: float
    """Mean pass rate across all items for the baseline."""
    cand_rate: float
    """Mean pass rate across all items for the candidate."""
    delta: float
    """cand_rate minus base_rate."""
    delta_ci: tuple[float, float]
    """Paired bootstrap 95 percent confidence interval for delta."""
    mcnemar_p: float | None
    """Exact McNemar p-value on majority-vote pass per item."""
    nfr_single_shot: float
    """Negative flips on sample index 0 divided by n_items."""
    nfr_single_shot_conditional: float
    """Same as nfr_single_shot, restricted to items the baseline got right on sample index 0."""
    reliable_nfr: float
    """Reliable regressions divided by n_items."""
    churn_rate: float
    """(Reliable regressions plus reliable improvements) divided by n_items."""
    flaky_rate: float
    inconclusive_rate: float
    slices: list[SliceSummary]
    calls_used: int
    est_cost_usd: float | None


class VerdictLevel(StrEnum):
    """Top-level verdict for a comparison run."""

    SHIP = "ship"
    SHIP_WITH_CAVEATS = "ship_with_caveats"
    BLOCK = "block"


class Reason(BaseModel):
    """One plain-English reason contributing to the verdict."""

    level: VerdictLevel
    message: str
    """Plain English, one sentence, saying what happened and what to fix."""
    case_ids: list[str] = []
    """IDs of the cases involved, when applicable."""


class Verdict(BaseModel):
    """Final ship or block decision for a comparison run."""

    level: VerdictLevel
    reasons: list[Reason]
    """Reasons sorted BLOCK first, then by number of cases."""


# ---------------------------------------------------------------------------
# Config models (SPEC section 4)
# ---------------------------------------------------------------------------


class SamplingConfig(BaseModel):
    """Controls how many samples are collected per test case."""

    mode: Literal["fixed", "adaptive"] = "adaptive"
    k: int = 5
    """Number of samples per version per case when mode is fixed."""
    k_min: int = 2
    """Minimum samples per version per case in adaptive mode."""
    k_max: int = 8
    """Maximum samples per version per case in adaptive mode."""
    step: int = 2
    """Extra samples added per unsettled item per adaptive round."""
    budget_calls: int = 1500
    """Hard cap on total model calls including judge calls."""


class GateConfig(BaseModel):
    """Thresholds that control when a run is blocked or flagged."""

    alpha: float = 0.05
    """Significance level for per-item Fisher tests after BH adjustment."""
    max_reliable_nfr: float = 0.03
    """Maximum allowed reliable negative flip rate before blocking."""
    block_on_critical_regression: bool = True
    """When True, any reliable regression on a critical case triggers BLOCK."""
    max_flaky_rate: float = 0.15
    """Above this flaky rate the verdict is SHIP_WITH_CAVEATS."""
    max_inconclusive_rate: float = 0.10
    """Above this inconclusive rate the verdict is SHIP_WITH_CAVEATS."""
    max_slice_drop: float = 0.10
    """A confident slice with delta below negative max_slice_drop triggers SHIP_WITH_CAVEATS."""


class RateLimitsConfig(BaseModel):
    """Requests-per-minute limits keyed by LiteLLM provider prefix."""

    groq: int = 30
    gemini: int = 10
    ollama: int = 0
    """0 means no limit."""


class ProvidersConfig(BaseModel):
    """Controls concurrency and rate limits for model calls."""

    rate_limits: RateLimitsConfig = RateLimitsConfig()
    concurrency: int = 4
    """Maximum number of in-flight model calls at once."""


class JudgeConfig(BaseModel):
    """Configuration for the LLM judge used by llm_judge scorers."""

    model: str = "gemini/gemini-2.5-flash"
    params: dict[str, Any] = {"temperature": 0}


class ProjectConfig(BaseModel):
    """Top-level config parsed from flipcheck.yaml."""

    project: str
    suite: str
    """Path to the JSONL test suite file."""
    default_scorers: list[ScorerConfig] = []
    """Scorers applied to every case that has no per-case scorers."""
    targets: dict[str, TargetConfig]
    """Named version targets, typically baseline and candidate."""
    judge: JudgeConfig = JudgeConfig()
    sampling: SamplingConfig = SamplingConfig()
    gates: GateConfig = GateConfig()
    providers: ProvidersConfig = ProvidersConfig()
    cache_path: str = ".flipcheck/flipcheck.sqlite"
