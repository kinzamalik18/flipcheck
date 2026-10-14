"""Tests for flipcheck/gate.py - verdict logic.

All tests are derived from SPEC section 10. The gate module already exists
as a partial stub; these tests check the full spec behavior.

One test per SPEC rule (section 10).
"""

from __future__ import annotations

import pytest

from flipcheck.types import (
    GateConfig,
    ItemClass,
    ItemCounts,
    ItemResult,
    RunSummary,
    SliceSummary,
    VerdictLevel,
)

# Skip the whole module gracefully when the implementation is not yet built.
try:
    import flipcheck.gate as _gate_mod

    decide = _gate_mod.decide
except AttributeError as _e:
    pytest.skip(f"flipcheck.gate not yet implemented: {_e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _counts(
    case_id: str,
    base_pass: int = 5,
    base_n: int = 5,
    cand_pass: int = 5,
    cand_n: int = 5,
) -> ItemCounts:
    return ItemCounts(
        case_id=case_id,
        base_pass=base_pass,
        base_n=base_n,
        cand_pass=cand_pass,
        cand_n=cand_n,
        base_first_pass=None,
        cand_first_pass=None,
    )


def _result(counts: ItemCounts, classification: ItemClass) -> ItemResult:
    return ItemResult(
        counts=counts,
        classification=classification,
        p_regression=None,
        p_improvement=None,
    )


def _summary(
    n_items: int = 5,
    base_rate: float = 0.8,
    cand_rate: float = 0.8,
    delta: float = 0.0,
    delta_ci: tuple[float, float] = (-0.05, 0.05),
    mcnemar_p: float | None = 1.0,
    nfr_single_shot: float = 0.0,
    nfr_single_shot_conditional: float = 0.0,
    reliable_nfr: float = 0.0,
    churn_rate: float = 0.0,
    flaky_rate: float = 0.0,
    inconclusive_rate: float = 0.0,
    slices: list[SliceSummary] | None = None,
    calls_used: int = 50,
    est_cost_usd: float | None = None,
) -> RunSummary:
    return RunSummary(
        n_items=n_items,
        base_rate=base_rate,
        cand_rate=cand_rate,
        delta=delta,
        delta_ci=delta_ci,
        mcnemar_p=mcnemar_p,
        nfr_single_shot=nfr_single_shot,
        nfr_single_shot_conditional=nfr_single_shot_conditional,
        reliable_nfr=reliable_nfr,
        churn_rate=churn_rate,
        flaky_rate=flaky_rate,
        inconclusive_rate=inconclusive_rate,
        slices=slices if slices is not None else [],
        calls_used=calls_used,
        est_cost_usd=est_cost_usd,
    )


def _gates(**kwargs) -> GateConfig:
    defaults = dict(
        alpha=0.05,
        max_reliable_nfr=0.03,
        block_on_critical_regression=True,
        max_flaky_rate=0.15,
        max_inconclusive_rate=0.10,
        max_slice_drop=0.10,
    )
    defaults.update(kwargs)
    return GateConfig(**defaults)


def _slice(
    tag: str,
    n_items: int = 10,
    base_rate: float = 0.8,
    cand_rate: float = 0.8,
    delta: float = 0.0,
    regressions: int = 0,
    improvements: int = 0,
    low_confidence: bool = False,
) -> SliceSummary:
    return SliceSummary(
        tag=tag,
        n_items=n_items,
        base_rate=base_rate,
        cand_rate=cand_rate,
        delta=delta,
        regressions=regressions,
        improvements=improvements,
        low_confidence=low_confidence,
    )


# ---------------------------------------------------------------------------
# Tests: BLOCK conditions
# ---------------------------------------------------------------------------


def test_ship_when_no_regressions_and_good_stats():
    """No regressions, CI straddles 0, low rates -> SHIP."""
    summary = _summary(reliable_nfr=0.0, delta_ci=(-0.02, 0.03))
    items = [
        _result(_counts("t1"), ItemClass.UNCHANGED_PASS),
        _result(_counts("t2"), ItemClass.UNCHANGED_PASS),
    ]
    verdict = decide(summary, items, critical=set(), gates=_gates())
    assert verdict.level == VerdictLevel.SHIP


def test_block_on_critical_regression_when_flag_is_true():
    """A critical case with RELIABLE_REGRESSION and block_on_critical_regression=True -> BLOCK."""
    summary = _summary(reliable_nfr=0.2)
    items = [
        _result(_counts("critical1", 5, 5, 0, 5), ItemClass.RELIABLE_REGRESSION),
    ]
    verdict = decide(
        summary,
        items,
        critical={"critical1"},
        gates=_gates(block_on_critical_regression=True),
    )
    assert verdict.level == VerdictLevel.BLOCK


def test_ship_with_caveats_on_critical_regression_when_flag_is_false():
    """A critical RELIABLE_REGRESSION with block_on_critical_regression=False -> SHIP_WITH_CAVEATS.

    The regression still exists; it just does not BLOCK. Since it is a regression,
    the run cannot SHIP cleanly.
    """
    summary = _summary(reliable_nfr=0.0, delta_ci=(-0.02, 0.03))
    items = [
        _result(_counts("critical1", 5, 5, 0, 5), ItemClass.RELIABLE_REGRESSION),
    ]
    verdict = decide(
        summary,
        items,
        critical={"critical1"},
        gates=_gates(block_on_critical_regression=False),
    )
    # Should not be BLOCK since the flag is False.
    assert verdict.level != VerdictLevel.BLOCK
    # There is a regression; it should surface as at least SHIP_WITH_CAVEATS.
    assert verdict.level == VerdictLevel.SHIP_WITH_CAVEATS


def test_block_when_reliable_nfr_exceeds_max():
    """reliable_nfr > max_reliable_nfr -> BLOCK."""
    summary = _summary(reliable_nfr=0.10, delta_ci=(-0.02, 0.03))
    items = []
    verdict = decide(summary, items, critical=set(), gates=_gates(max_reliable_nfr=0.03))
    assert verdict.level == VerdictLevel.BLOCK


def test_block_when_delta_ci_upper_bound_below_zero():
    """delta_ci upper bound < 0 (overall quality dropped) -> BLOCK."""
    summary = _summary(delta=-0.15, delta_ci=(-0.25, -0.05))
    items = []
    verdict = decide(summary, items, critical=set(), gates=_gates())
    assert verdict.level == VerdictLevel.BLOCK


# ---------------------------------------------------------------------------
# Tests: SHIP_WITH_CAVEATS conditions
# ---------------------------------------------------------------------------


def test_ship_with_caveats_on_non_critical_reliable_regression():
    """A non-critical RELIABLE_REGRESSION item with no other problems -> SHIP_WITH_CAVEATS."""
    summary = _summary(reliable_nfr=0.0, delta_ci=(-0.02, 0.03))
    items = [
        _result(_counts("nc1", 5, 5, 0, 5), ItemClass.RELIABLE_REGRESSION),
    ]
    verdict = decide(summary, items, critical=set(), gates=_gates())
    assert verdict.level == VerdictLevel.SHIP_WITH_CAVEATS


def test_ship_with_caveats_when_flaky_rate_exceeds_max():
    """flaky_rate > max_flaky_rate -> SHIP_WITH_CAVEATS."""
    summary = _summary(flaky_rate=0.20, delta_ci=(-0.02, 0.05))
    items = []
    verdict = decide(summary, items, critical=set(), gates=_gates(max_flaky_rate=0.15))
    assert verdict.level == VerdictLevel.SHIP_WITH_CAVEATS


def test_ship_with_caveats_when_inconclusive_rate_exceeds_max():
    """inconclusive_rate > max_inconclusive_rate -> SHIP_WITH_CAVEATS."""
    summary = _summary(inconclusive_rate=0.15, delta_ci=(-0.02, 0.05))
    items = []
    verdict = decide(summary, items, critical=set(), gates=_gates(max_inconclusive_rate=0.10))
    assert verdict.level == VerdictLevel.SHIP_WITH_CAVEATS


def test_ship_with_caveats_when_confident_slice_has_large_negative_delta():
    """A slice with delta < -max_slice_drop and low_confidence=False -> SHIP_WITH_CAVEATS."""
    bad_slice = _slice("urdu", delta=-0.20, low_confidence=False)
    summary = _summary(slices=[bad_slice], delta_ci=(-0.02, 0.05))
    items = []
    verdict = decide(summary, items, critical=set(), gates=_gates(max_slice_drop=0.10))
    assert verdict.level == VerdictLevel.SHIP_WITH_CAVEATS


def test_ship_not_caveats_when_low_confidence_slice_has_large_negative_delta():
    """A low-confidence slice with delta < -max_slice_drop does NOT trigger SHIP_WITH_CAVEATS.

    Spec: 'a slice that is not low confidence has delta < -max_slice_drop'.
    Low-confidence slices are excluded from this rule.
    """
    bad_slice = _slice("rare-tag", delta=-0.20, low_confidence=True)
    summary = _summary(slices=[bad_slice], delta_ci=(-0.02, 0.05))
    items = []
    verdict = decide(summary, items, critical=set(), gates=_gates(max_slice_drop=0.10))
    assert verdict.level == VerdictLevel.SHIP


# ---------------------------------------------------------------------------
# Tests: priority and reason structure
# ---------------------------------------------------------------------------


def test_multiple_block_conditions_still_gives_block():
    """Even with multiple BLOCK conditions, level is BLOCK (not SHIP_WITH_CAVEATS)."""
    summary = _summary(reliable_nfr=0.10, delta_ci=(-0.25, -0.05))
    items = [
        _result(_counts("c1", 5, 5, 0, 5), ItemClass.RELIABLE_REGRESSION),
    ]
    verdict = decide(summary, items, critical={"c1"}, gates=_gates())
    assert verdict.level == VerdictLevel.BLOCK


def test_block_wins_over_ship_with_caveats_when_both_conditions_present():
    """When both BLOCK and SHIP_WITH_CAVEATS conditions are met, final level is BLOCK."""
    summary = _summary(reliable_nfr=0.10, flaky_rate=0.20, delta_ci=(-0.02, 0.05))
    items = [
        _result(_counts("nc1", 5, 5, 0, 5), ItemClass.RELIABLE_REGRESSION),
    ]
    verdict = decide(summary, items, critical=set(), gates=_gates(max_reliable_nfr=0.03))
    assert verdict.level == VerdictLevel.BLOCK


def test_reasons_sorted_block_first():
    """Reasons list should have BLOCK reasons before SHIP_WITH_CAVEATS reasons."""
    summary = _summary(reliable_nfr=0.10, flaky_rate=0.20, delta_ci=(-0.02, 0.05))
    items = [
        _result(_counts("nc1", 5, 5, 0, 5), ItemClass.RELIABLE_REGRESSION),
        _result(_counts("nc2", 5, 5, 0, 5), ItemClass.RELIABLE_REGRESSION),
    ]
    verdict = decide(summary, items, critical=set(), gates=_gates(max_reliable_nfr=0.03))
    assert verdict.level == VerdictLevel.BLOCK
    block_reasons = [r for r in verdict.reasons if r.level == VerdictLevel.BLOCK]
    caveats_reasons = [r for r in verdict.reasons if r.level == VerdictLevel.SHIP_WITH_CAVEATS]
    if block_reasons and caveats_reasons:
        # All BLOCK reasons should appear before any SHIP_WITH_CAVEATS reasons.
        last_block_idx = max(verdict.reasons.index(r) for r in block_reasons)
        first_caveat_idx = min(verdict.reasons.index(r) for r in caveats_reasons)
        assert last_block_idx < first_caveat_idx


def test_no_ship_with_caveats_level_when_block_condition_fires():
    """The final level should be BLOCK, not SHIP_WITH_CAVEATS, when a BLOCK condition fires."""
    summary = _summary(reliable_nfr=0.0, delta_ci=(-0.25, -0.05))
    items = [
        _result(_counts("c1", 5, 5, 0, 5), ItemClass.RELIABLE_REGRESSION),
    ]
    verdict = decide(
        summary, items, critical={"c1"}, gates=_gates(block_on_critical_regression=True)
    )
    assert verdict.level == VerdictLevel.BLOCK


def test_reasons_list_is_nonempty_when_not_ship():
    """When verdict is not SHIP, there should be at least one reason."""
    summary = _summary(reliable_nfr=0.10, delta_ci=(-0.02, 0.05))
    items = []
    verdict = decide(summary, items, critical=set(), gates=_gates(max_reliable_nfr=0.03))
    assert verdict.level == VerdictLevel.BLOCK
    assert len(verdict.reasons) >= 1


def test_block_reason_contains_case_ids_for_critical_regression():
    """When a critical regression triggers BLOCK, the reason should include the case_id."""
    summary = _summary(reliable_nfr=0.0, delta_ci=(-0.02, 0.05))
    items = [
        _result(_counts("my-critical", 5, 5, 0, 5), ItemClass.RELIABLE_REGRESSION),
    ]
    verdict = decide(summary, items, critical={"my-critical"}, gates=_gates())
    block_reasons = [r for r in verdict.reasons if r.level == VerdictLevel.BLOCK]
    assert any("my-critical" in r.case_ids for r in block_reasons)


def test_no_non_ship_reasons_when_verdict_is_ship():
    """When verdict is SHIP, all reasons (if any) must have level=SHIP (no triggered rules)."""
    summary = _summary(reliable_nfr=0.0, delta_ci=(-0.02, 0.05))
    items = [
        _result(_counts("t1"), ItemClass.UNCHANGED_PASS),
    ]
    verdict = decide(summary, items, critical=set(), gates=_gates())
    assert verdict.level == VerdictLevel.SHIP
    non_ship_reasons = [
        r for r in verdict.reasons if r.level != VerdictLevel.SHIP
    ]
    assert non_ship_reasons == []


def test_delta_ci_upper_exactly_zero_is_not_block():
    """delta_ci upper bound == 0: spec says 'below 0' triggers BLOCK, so 0 itself does not.

    An upper bound of exactly 0 should NOT trigger BLOCK.
    """
    summary = _summary(reliable_nfr=0.0, delta_ci=(-0.05, 0.0))
    items = []
    verdict = decide(summary, items, critical=set(), gates=_gates())
    assert verdict.level != VerdictLevel.BLOCK


def test_reliable_nfr_exactly_at_max_is_not_block():
    """reliable_nfr == max_reliable_nfr (not strictly greater) should not trigger BLOCK."""
    summary = _summary(reliable_nfr=0.03, delta_ci=(-0.02, 0.05))
    items = []
    verdict = decide(summary, items, critical=set(), gates=_gates(max_reliable_nfr=0.03))
    assert verdict.level != VerdictLevel.BLOCK


def test_flaky_rate_exactly_at_max_is_not_caveats():
    """flaky_rate == max_flaky_rate (not strictly greater) should not trigger SHIP_WITH_CAVEATS."""
    summary = _summary(flaky_rate=0.15, delta_ci=(-0.02, 0.05))
    items = []
    verdict = decide(summary, items, critical=set(), gates=_gates(max_flaky_rate=0.15))
    assert verdict.level == VerdictLevel.SHIP


def test_inconclusive_rate_exactly_at_max_is_not_caveats():
    """inconclusive_rate == max_inconclusive_rate (not strictly greater) should not trigger SHIP_WITH_CAVEATS."""
    summary = _summary(inconclusive_rate=0.10, delta_ci=(-0.02, 0.05))
    items = []
    verdict = decide(summary, items, critical=set(), gates=_gates(max_inconclusive_rate=0.10))
    assert verdict.level == VerdictLevel.SHIP
