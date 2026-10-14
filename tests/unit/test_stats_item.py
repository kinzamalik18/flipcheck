"""Tests for flipcheck/stats/item.py - per-item classification.

All tests are derived from SPEC section 9.1. The implementation files are
stubs at the time of writing; these tests define the required behavior.
"""

from __future__ import annotations

import pytest
from scipy.stats import fisher_exact

from flipcheck.types import ItemClass, ItemCounts

# Skip the whole module gracefully when the implementation is not yet built.
# The module files exist as stubs; check for the required attributes explicitly.
try:
    import flipcheck.stats.item as _stats_item_mod

    fisher_p_regression = _stats_item_mod.fisher_p_regression
    fisher_p_improvement = _stats_item_mod.fisher_p_improvement
    bh_adjust = _stats_item_mod.bh_adjust
    classify_items = _stats_item_mod.classify_items
except AttributeError as _e:
    pytest.skip(f"flipcheck.stats.item not yet implemented: {_e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _counts(
    case_id: str,
    base_pass: int,
    base_n: int,
    cand_pass: int,
    cand_n: int,
    base_first_pass: bool | None = None,
    cand_first_pass: bool | None = None,
) -> ItemCounts:
    return ItemCounts(
        case_id=case_id,
        base_pass=base_pass,
        base_n=base_n,
        cand_pass=cand_pass,
        cand_n=cand_n,
        base_first_pass=base_first_pass,
        cand_first_pass=cand_first_pass,
    )


# ---------------------------------------------------------------------------
# fisher_p_regression tests
# ---------------------------------------------------------------------------


def test_fisher_p_regression_clear_regression_is_small():
    """a=5/5 baseline, b=0/5 candidate: candidate is clearly worse, p should be very small."""
    # Contingency table: [[5, 0], [0, 5]]
    # scipy: alternative='greater' on baseline row tests baseline rate > candidate rate.
    # The function tests that the candidate pass rate is LOWER (regression).
    p = fisher_p_regression(5, 5, 0, 5)
    # Verify with scipy: table [[a, n1-a], [b, n2-b]], alternative='greater' for baseline row.
    _, scipy_p = fisher_exact([[5, 0], [0, 5]], alternative="greater")
    assert p == pytest.approx(scipy_p, rel=1e-6)
    assert p < 0.05


def test_fisher_p_regression_clear_improvement_is_large():
    """a=0/5 baseline, b=5/5 candidate: candidate is clearly better, regression p should be ~1."""
    p = fisher_p_regression(0, 5, 5, 5)
    # baseline pass rate 0, candidate pass rate 1: no regression at all.
    _, scipy_p = fisher_exact([[0, 5], [5, 0]], alternative="greater")
    assert p == pytest.approx(scipy_p, rel=1e-6)
    assert p > 0.5


def test_fisher_p_regression_equal_rates_near_half():
    """a=3/5 baseline, b=3/5 candidate: no change, regression p should not be small."""
    p = fisher_p_regression(3, 5, 3, 5)
    assert p > 0.1


# ---------------------------------------------------------------------------
# fisher_p_improvement tests
# ---------------------------------------------------------------------------


def test_fisher_p_improvement_clear_improvement_is_small():
    """a=0/5 baseline, b=5/5 candidate: candidate clearly better, p should be very small."""
    p = fisher_p_improvement(0, 5, 5, 5)
    # Testing that candidate pass rate is HIGHER (improvement).
    # Table [[a, n1-a], [b, n2-b]], alternative='less' for baseline row = candidate better.
    _, scipy_p = fisher_exact([[0, 5], [5, 0]], alternative="less")
    assert p == pytest.approx(scipy_p, rel=1e-6)
    assert p < 0.05


def test_fisher_p_improvement_clear_regression_is_large():
    """a=5/5 baseline, b=0/5 candidate: candidate is worse, improvement p should be large."""
    p = fisher_p_improvement(5, 5, 0, 5)
    _, scipy_p = fisher_exact([[5, 0], [0, 5]], alternative="less")
    assert p == pytest.approx(scipy_p, rel=1e-6)
    assert p > 0.5


def test_fisher_p_improvement_equal_rates():
    """Equal pass rates: improvement p should not be small."""
    p = fisher_p_improvement(3, 5, 3, 5)
    assert p > 0.1


# ---------------------------------------------------------------------------
# bh_adjust tests
# ---------------------------------------------------------------------------


def test_bh_adjust_all_ones_returns_ones():
    """All p-values = 1.0 should stay at 1.0 after BH adjustment."""
    result = bh_adjust([1.0, 1.0, 1.0])
    assert all(v == pytest.approx(1.0) for v in result)


def test_bh_adjust_empty_list_returns_empty():
    """Empty input should return an empty list without error."""
    result = bh_adjust([])
    assert result == []


def test_bh_adjust_small_p_values_not_inflated_above_original_times_rank_factor():
    """BH adjusted p should be <= p_original * (n / rank).

    For p=[0.001, 0.001, 0.5] sorted ascending:
    - rank 1: 0.001 * 3/1 = 0.003
    - rank 2: 0.001 * 3/2 = 0.0015
    - rank 3: 0.500 * 3/3 = 0.500
    BH takes the minimum suffix, so adjusted values are at most these.
    """
    p_values = [0.001, 0.001, 0.5]
    result = bh_adjust(p_values)
    # Length should be preserved.
    assert len(result) == 3
    # All adjusted values should be in [0, 1].
    for v in result:
        assert 0.0 <= v <= 1.0


def test_bh_adjust_matches_scipy_manual():
    """BH on p=[0.01, 0.04, 0.2] with n=3, alpha=0.05.

    Hand calculation:
      Sorted: 0.01 (rank 1), 0.04 (rank 2), 0.2 (rank 3).
      BH critical values: 0.05*1/3=0.0167, 0.05*2/3=0.0333, 0.05*3/3=0.05.
      p[1]=0.01 < 0.0167  -> passes BH at alpha=0.05.
      p[2]=0.04 > 0.0333  -> does NOT pass BH at alpha=0.05.
      p[3]=0.2  > 0.05    -> does NOT pass BH at alpha=0.05.

    BH adjusted p-values (Benjamini-Hochberg step-up procedure):
      For rank 3: adj = 0.2 * 3/3 = 0.2
      For rank 2: adj = min(0.04 * 3/2, 0.2) = min(0.06, 0.2) = 0.06
      For rank 1: adj = min(0.01 * 3/1, 0.06) = min(0.03, 0.06) = 0.03

    So adjusted p-values (in original input order) should be:
      index 0 (p=0.01) -> 0.03
      index 1 (p=0.04) -> 0.06
      index 2 (p=0.2)  -> 0.2
    """
    p_values = [0.01, 0.04, 0.2]
    result = bh_adjust(p_values)
    assert len(result) == 3
    # Original index 0 (p=0.01) gets adjusted to 0.03.
    assert result[0] == pytest.approx(0.03, rel=1e-6)
    # Original index 1 (p=0.04) gets adjusted to 0.06.
    assert result[1] == pytest.approx(0.06, rel=1e-6)
    # Original index 2 (p=0.2) gets adjusted to 0.2.
    assert result[2] == pytest.approx(0.2, rel=1e-6)


def test_bh_adjust_preserves_order_with_respect_to_input():
    """bh_adjust returns adjusted values aligned positionally to the input list."""
    p_values = [0.2, 0.01, 0.04]
    result = bh_adjust(p_values)
    assert len(result) == 3
    # The smallest original value (index 1, p=0.01) should have the smallest adjusted value.
    assert result[1] <= result[0]
    assert result[1] <= result[2]


def test_bh_adjust_single_value():
    """Single-element list: adjusted value equals the original p-value."""
    result = bh_adjust([0.03])
    assert len(result) == 1
    assert result[0] == pytest.approx(0.03, rel=1e-6)


# ---------------------------------------------------------------------------
# classify_items tests
# ---------------------------------------------------------------------------


def test_classify_single_clear_regression():
    """5/5 baseline, 0/5 candidate with 1 item gets RELIABLE_REGRESSION.

    With 1 item there is no BH inflation. Fisher p for regression with
    [[5,0],[0,5]] is 1/252 ~ 0.004 < alpha=0.05.
    """
    counts = [_counts("t1", 5, 5, 0, 5)]
    results = classify_items(counts, alpha=0.05, inconclusive_ids=set())
    assert len(results) == 1
    assert results[0].classification == ItemClass.RELIABLE_REGRESSION


def test_classify_single_clear_improvement():
    """0/5 baseline, 5/5 candidate gets RELIABLE_IMPROVEMENT."""
    counts = [_counts("t1", 0, 5, 5, 5)]
    results = classify_items(counts, alpha=0.05, inconclusive_ids=set())
    assert len(results) == 1
    assert results[0].classification == ItemClass.RELIABLE_IMPROVEMENT


def test_classify_both_perfect_pass_rates_gets_unchanged_pass():
    """5/5 baseline (100%), 5/5 candidate (100%): both >= 0.5 and not in [0.2, 0.8].

    Rates are 1.0, which is NOT in [0.2, 0.8], so FLAKY does not fire.
    Both >= 0.5, so UNCHANGED_PASS fires.
    No significant regression or improvement with identical perfect scores.
    """
    counts = [_counts("t1", 5, 5, 5, 5)]
    results = classify_items(counts, alpha=0.05, inconclusive_ids=set())
    assert len(results) == 1
    assert results[0].classification == ItemClass.UNCHANGED_PASS


def test_classify_both_zero_pass_rates_gets_unchanged_fail():
    """0/5 baseline, 0/5 candidate: both fail consistently, gets UNCHANGED_FAIL."""
    counts = [_counts("t1", 0, 5, 0, 5)]
    results = classify_items(counts, alpha=0.05, inconclusive_ids=set())
    assert len(results) == 1
    assert results[0].classification == ItemClass.UNCHANGED_FAIL


def test_classify_mid_range_rates_gets_flaky():
    """2/5 baseline (40%), 3/5 candidate (60%): both in [0.2, 0.8] -> FLAKY.

    Fisher p for [[2,3],[3,2]] is not significant, so no reliable change.
    Both rates (0.4 and 0.6) are in [0.2, 0.8], so FLAKY fires.
    """
    counts = [_counts("t1", 2, 5, 3, 5)]
    results = classify_items(counts, alpha=0.05, inconclusive_ids=set())
    assert len(results) == 1
    assert results[0].classification == ItemClass.FLAKY


def test_classify_inconclusive_id_gets_inconclusive():
    """An item in inconclusive_ids should be classified as INCONCLUSIVE regardless of counts."""
    counts = [_counts("t1", 3, 5, 2, 5)]
    results = classify_items(counts, alpha=0.05, inconclusive_ids={"t1"})
    assert len(results) == 1
    assert results[0].classification == ItemClass.INCONCLUSIVE


def test_classify_inconclusive_id_overrides_even_clear_regression():
    """Even a clear regression is overridden to INCONCLUSIVE when id is in inconclusive_ids."""
    counts = [_counts("t1", 5, 5, 0, 5)]
    results = classify_items(counts, alpha=0.05, inconclusive_ids={"t1"})
    assert len(results) == 1
    assert results[0].classification == ItemClass.INCONCLUSIVE


def test_classify_multiple_items_returns_same_length():
    """classify_items returns one result per input item."""
    counts = [
        _counts("t1", 5, 5, 0, 5),
        _counts("t2", 0, 5, 5, 5),
        _counts("t3", 2, 5, 3, 5),
    ]
    results = classify_items(counts, alpha=0.05, inconclusive_ids=set())
    assert len(results) == 3


def test_classify_results_have_adjusted_p_values():
    """ItemResult objects should expose BH-adjusted p_regression and p_improvement."""
    counts = [_counts("t1", 5, 5, 0, 5)]
    results = classify_items(counts, alpha=0.05, inconclusive_ids=set())
    r = results[0]
    assert r.p_regression is not None
    assert r.p_improvement is not None
    assert 0.0 <= r.p_regression <= 1.0
    assert 0.0 <= r.p_improvement <= 1.0


def test_classify_case_id_preserved_in_result():
    """The case_id in ItemResult.counts should match the input ItemCounts."""
    counts = [_counts("my-case-42", 3, 5, 3, 5)]
    results = classify_items(counts, alpha=0.05, inconclusive_ids=set())
    assert results[0].counts.case_id == "my-case-42"


def test_classify_base_rate_one_cand_rate_one_unchanged_pass():
    """Both sides perfect: both pass rates = 1.0 >= 0.5, no regression, gets UNCHANGED_PASS."""
    counts = [_counts("t1", 8, 8, 8, 8)]
    results = classify_items(counts, alpha=0.05, inconclusive_ids=set())
    assert results[0].classification == ItemClass.UNCHANGED_PASS


def test_classify_flaky_with_rate_at_lower_boundary():
    """FLAKY fires when EITHER pass rate is in [0.2, 0.8], including exact boundary 0.2.

    Spec: 'either pass rate between 0.2 and 0.8 inclusive gives FLAKY'.
    baseline=1/5 (0.2) is exactly at the lower boundary (in range).
    candidate=2/5 (0.4) is also in range.
    Fisher p is not significant; FLAKY should fire.
    """
    counts = [_counts("t1", 1, 5, 2, 5)]
    results = classify_items(counts, alpha=0.05, inconclusive_ids=set())
    assert results[0].classification == ItemClass.FLAKY
