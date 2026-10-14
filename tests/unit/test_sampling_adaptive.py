"""Tests for flipcheck/sampling/adaptive.py - adaptive sampling policy.

All tests are derived from SPEC section 8.2. The implementation is a stub
at the time of writing; these tests define the required behavior.

The pure function next_round(counts, cfg, budget_left, critical) -> dict[str, int]
returns {case_id: extra_samples_per_version} for the next sampling round.
An empty dict means stop.
"""

from __future__ import annotations

import pytest

from flipcheck.types import ItemCounts, SamplingConfig

# Skip the whole module gracefully when the implementation is not yet built.
try:
    import flipcheck.sampling.adaptive as _adaptive_mod

    next_round = _adaptive_mod.next_round
except AttributeError as _e:
    pytest.skip(
        f"flipcheck.sampling.adaptive not yet implemented: {_e}", allow_module_level=True
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(k_min: int = 2, k_max: int = 8, step: int = 2, budget_calls: int = 1500) -> SamplingConfig:
    return SamplingConfig(
        mode="adaptive",
        k_min=k_min,
        k_max=k_max,
        step=step,
        budget_calls=budget_calls,
    )


def _counts(
    case_id: str,
    base_pass: int,
    base_n: int,
    cand_pass: int,
    cand_n: int,
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


# ---------------------------------------------------------------------------
# Settlement helpers.
# An item is unanimously settled when both versions are unanimous.
# Both versions agree on all-pass or all-fail.
# ---------------------------------------------------------------------------


def _settled_pass(case_id: str, n: int = 5) -> ItemCounts:
    """Both versions all-pass (unanimous, settled)."""
    return _counts(case_id, n, n, n, n)


def _settled_fail(case_id: str, n: int = 5) -> ItemCounts:
    """Both versions all-fail (unanimous, settled)."""
    return _counts(case_id, 0, n, 0, n)


def _unsettled(case_id: str, base_pass: int = 2, n: int = 4) -> ItemCounts:
    """Neither unanimous nor all opposite; neither side settled."""
    return _counts(case_id, base_pass, n, n - base_pass, n)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_settled_returns_empty_dict():
    """When all items are settled (unanimous agree), next_round should return {}."""
    cfg = _cfg(k_max=8, step=2)
    counts = [_settled_pass("t1"), _settled_pass("t2"), _settled_fail("t3")]
    result = next_round(counts, cfg, budget_left=100, critical=set())
    assert result == {}


def test_unsettled_item_below_k_max_returns_step_samples():
    """An unsettled item below k_max gets step extra samples per version."""
    cfg = _cfg(k_max=8, step=2)
    counts = [_unsettled("t1", base_pass=2, n=4)]
    result = next_round(counts, cfg, budget_left=100, critical=set())
    assert "t1" in result
    assert result["t1"] == cfg.step


def test_item_at_k_max_not_returned():
    """An item that has already reached k_max should not be included in next_round."""
    cfg = _cfg(k_max=5, step=2)
    # base_n == k_max means this item is capped.
    counts = [_counts("t1", 2, 5, 3, 5)]
    result = next_round(counts, cfg, budget_left=100, critical=set())
    assert "t1" not in result


def test_budget_zero_returns_empty_dict():
    """With budget_left=0, no samples can be added; next_round returns {}."""
    cfg = _cfg(k_max=8, step=2)
    counts = [_unsettled("t1")]
    result = next_round(counts, cfg, budget_left=0, critical=set())
    assert result == {}


def test_critical_item_prioritized_over_noncritical_under_tight_budget():
    """Under tight budget, critical items are scheduled before non-critical.

    Budget is enough for only one item (step=2 calls per version = 4 total).
    t1 is non-critical, t2 is critical. t2 should be picked.
    """
    cfg = _cfg(k_max=8, step=2)
    counts = [
        _unsettled("t1", base_pass=2, n=4),
        _unsettled("t2", base_pass=2, n=4),
    ]
    # Budget exactly covers one item: step * 2 versions = 4.
    result = next_round(counts, cfg, budget_left=4, critical={"t2"})
    assert "t2" in result
    assert "t1" not in result


def test_larger_rate_difference_prioritized_under_tight_budget():
    """Under tight budget, items with larger current rate difference come first.

    t1: base 2/4 = 0.5, cand 2/4 = 0.5 -> diff = 0.0
    t2: base 1/4 = 0.25, cand 3/4 = 0.75 -> diff = 0.5
    Budget for only one item. t2 should be picked.
    """
    cfg = _cfg(k_max=8, step=2)
    counts = [
        _counts("t1", 2, 4, 2, 4),
        _counts("t2", 1, 4, 3, 4),
    ]
    result = next_round(counts, cfg, budget_left=4, critical=set())
    assert "t2" in result
    assert "t1" not in result


def test_budget_limits_number_of_items_scheduled():
    """With budget covering exactly 2 items and 3 unsettled, only 2 are included.

    Each item costs step * 2 = 2 * 2 = 4 calls (step samples per version, 2 versions).
    Budget = 8 covers exactly 2 items.
    """
    cfg = _cfg(k_max=8, step=2)
    counts = [
        _unsettled("t1", base_pass=2, n=4),
        _unsettled("t2", base_pass=2, n=4),
        _unsettled("t3", base_pass=2, n=4),
    ]
    result = next_round(counts, cfg, budget_left=8, critical=set())
    assert len(result) == 2


def test_case_with_base_n_zero_gets_included_when_budget_allows():
    """A case with base_n=0 (hasn't started) should be included if budget allows."""
    cfg = _cfg(k_min=2, k_max=8, step=2)
    counts = [_counts("t1", 0, 0, 0, 0)]
    result = next_round(counts, cfg, budget_left=100, critical=set())
    assert "t1" in result


def test_result_only_contains_case_ids_below_k_max():
    """next_round should never return a case_id with base_n >= k_max."""
    cfg = _cfg(k_max=5, step=2)
    counts = [
        _counts("at_max", 2, 5, 3, 5),
        _counts("below_max", 2, 3, 3, 3),
    ]
    result = next_round(counts, cfg, budget_left=100, critical=set())
    assert "at_max" not in result


def test_tie_in_rate_difference_broken_by_id():
    """When rate differences are equal and no critical tag, tie is broken by case_id alphabetically.

    b_item and a_item have equal rate differences. With budget for 1 item,
    the item with the lexicographically smaller id should be chosen.
    """
    cfg = _cfg(k_max=8, step=2)
    counts = [
        _counts("b_item", 1, 4, 3, 4),
        _counts("a_item", 1, 4, 3, 4),
    ]
    result = next_round(counts, cfg, budget_left=4, critical=set())
    assert "a_item" in result
    assert "b_item" not in result


def test_all_items_at_k_max_returns_empty():
    """If all items have base_n >= k_max, next_round returns {}."""
    cfg = _cfg(k_max=5, step=2)
    counts = [
        _counts("t1", 3, 5, 2, 5),
        _counts("t2", 5, 5, 5, 5),
    ]
    result = next_round(counts, cfg, budget_left=100, critical=set())
    assert result == {}


def test_empty_counts_returns_empty_dict():
    """With no items, next_round should return {} without error."""
    cfg = _cfg()
    result = next_round([], cfg, budget_left=100, critical=set())
    assert result == {}


def test_settled_and_unsettled_mix_only_unsettled_scheduled():
    """Only unsettled items below k_max should be in next_round result."""
    cfg = _cfg(k_max=8, step=2)
    counts = [
        _settled_pass("settled1"),
        _settled_fail("settled2"),
        _unsettled("unsettled1"),
    ]
    result = next_round(counts, cfg, budget_left=100, critical=set())
    assert "settled1" not in result
    assert "settled2" not in result
    assert "unsettled1" in result
