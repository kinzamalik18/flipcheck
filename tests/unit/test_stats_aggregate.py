"""Tests for flipcheck/stats/aggregate.py - run-level summary statistics.

All tests are derived from SPEC section 9.2. The implementation is a stub
at the time of writing; these tests define the required behavior.
"""

from __future__ import annotations

import pytest

from flipcheck.types import ItemClass, ItemCounts, ItemResult

# Skip the whole module gracefully when the implementation is not yet built.
try:
    import flipcheck.stats.aggregate as _agg_mod

    compute_summary = _agg_mod.compute_summary
except AttributeError as _e:
    pytest.skip(f"flipcheck.stats.aggregate not yet implemented: {_e}", allow_module_level=True)


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


def _result(counts: ItemCounts, classification: ItemClass) -> ItemResult:
    return ItemResult(
        counts=counts,
        classification=classification,
        p_regression=None,
        p_improvement=None,
    )


# ---------------------------------------------------------------------------
# Fixtures: three items with known values
# ---------------------------------------------------------------------------
#
# Item A: base 4/4 (rate=1.0), cand 4/4 (rate=1.0) -> UNCHANGED_PASS
#   base_first_pass=True, cand_first_pass=True
# Item B: base 4/4 (rate=1.0), cand 0/4 (rate=0.0) -> RELIABLE_REGRESSION
#   base_first_pass=True, cand_first_pass=False
# Item C: base 0/4 (rate=0.0), cand 4/4 (rate=1.0) -> RELIABLE_IMPROVEMENT
#   base_first_pass=False, cand_first_pass=True
#
# Expected aggregate values:
#   base_rate = mean(1.0, 1.0, 0.0) = 2/3 ~ 0.6667
#   cand_rate = mean(1.0, 0.0, 1.0) = 2/3 ~ 0.6667
#   delta = cand_rate - base_rate = 0.0
#   nfr_single_shot: negative flips on sample 0 = items where base_first_pass=True
#                    and cand_first_pass=False -> only item B -> 1/3 ~ 0.3333
#   nfr_single_shot_conditional: over items where base_first_pass=True (A and B):
#                    B is a negative flip -> 1/2 = 0.5
#   reliable_nfr = 1 RELIABLE_REGRESSION / 3 = 1/3 ~ 0.3333
#   churn_rate = (1 regression + 1 improvement) / 3 = 2/3 ~ 0.6667
#   flaky_rate = 0 FLAKY / 3 = 0.0
#   inconclusive_rate = 0 INCONCLUSIVE / 3 = 0.0
#   calls_used = (4+4)*3 = 24


@pytest.fixture()
def three_item_data():
    c_a = _counts("a", 4, 4, 4, 4, base_first_pass=True, cand_first_pass=True)
    c_b = _counts("b", 4, 4, 0, 4, base_first_pass=True, cand_first_pass=False)
    c_c = _counts("c", 0, 4, 4, 4, base_first_pass=False, cand_first_pass=True)
    r_a = _result(c_a, ItemClass.UNCHANGED_PASS)
    r_b = _result(c_b, ItemClass.RELIABLE_REGRESSION)
    r_c = _result(c_c, ItemClass.RELIABLE_IMPROVEMENT)
    return [c_a, c_b, c_c], [r_a, r_b, r_c]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_delta_equals_cand_rate_minus_base_rate(three_item_data):
    """delta should be cand_rate - base_rate."""
    counts, results = three_item_data
    summary = compute_summary(counts, results, seed=42)
    expected_delta = summary.cand_rate - summary.base_rate
    assert summary.delta == pytest.approx(expected_delta, abs=1e-9)


def test_delta_ci_is_tuple_of_two_floats(three_item_data):
    """delta_ci should be a tuple of exactly two floats."""
    counts, results = three_item_data
    summary = compute_summary(counts, results, seed=42)
    ci = summary.delta_ci
    assert isinstance(ci, tuple)
    assert len(ci) == 2
    assert isinstance(ci[0], float)
    assert isinstance(ci[1], float)


def test_delta_ci_contains_point_estimate(three_item_data):
    """The 95% CI should contain the point estimate delta."""
    counts, results = three_item_data
    summary = compute_summary(counts, results, seed=42)
    assert summary.delta_ci[0] <= summary.delta <= summary.delta_ci[1]


def test_delta_ci_reproducible_with_same_seed(three_item_data):
    """Calling with seed=42 twice should give identical CI bounds."""
    counts, results = three_item_data
    s1 = compute_summary(counts, results, seed=42)
    s2 = compute_summary(counts, results, seed=42)
    assert s1.delta_ci[0] == s2.delta_ci[0]
    assert s1.delta_ci[1] == s2.delta_ci[1]


def test_delta_ci_differs_with_different_seed(three_item_data):
    """Different seeds should (in general) produce different CI bounds."""
    counts, results = three_item_data
    s1 = compute_summary(counts, results, seed=42)
    s2 = compute_summary(counts, results, seed=99)
    # It is extremely unlikely that two different seeds give the exact same CI.
    assert s1.delta_ci != s2.delta_ci


def test_nfr_single_shot_counts_negative_flips_over_all_items(three_item_data):
    """nfr_single_shot = negative flips on sample 0 / n_items.

    Item B: base_first_pass=True, cand_first_pass=False -> 1 negative flip.
    n_items = 3 -> nfr_single_shot = 1/3.
    """
    counts, results = three_item_data
    summary = compute_summary(counts, results, seed=42)
    assert summary.nfr_single_shot == pytest.approx(1 / 3, rel=1e-6)


def test_nfr_single_shot_conditional_only_over_base_correct(three_item_data):
    """nfr_single_shot_conditional = negative flips / items where base_first_pass=True.

    Items where base_first_pass=True: A and B (2 items).
    Negative flip in that subset: only B.
    nfr_single_shot_conditional = 1/2 = 0.5.
    """
    counts, results = three_item_data
    summary = compute_summary(counts, results, seed=42)
    assert summary.nfr_single_shot_conditional == pytest.approx(0.5, rel=1e-6)


def test_reliable_nfr_is_regressions_over_n_items(three_item_data):
    """reliable_nfr = count(RELIABLE_REGRESSION) / n_items = 1/3."""
    counts, results = three_item_data
    summary = compute_summary(counts, results, seed=42)
    assert summary.reliable_nfr == pytest.approx(1 / 3, rel=1e-6)


def test_churn_rate_is_regressions_plus_improvements_over_n_items(three_item_data):
    """churn_rate = (1 regression + 1 improvement) / 3 = 2/3."""
    counts, results = three_item_data
    summary = compute_summary(counts, results, seed=42)
    assert summary.churn_rate == pytest.approx(2 / 3, rel=1e-6)


def test_flaky_rate_is_zero_when_no_flaky_items(three_item_data):
    """flaky_rate = 0 FLAKY items / 3 = 0.0."""
    counts, results = three_item_data
    summary = compute_summary(counts, results, seed=42)
    assert summary.flaky_rate == pytest.approx(0.0, abs=1e-9)


def test_inconclusive_rate_is_zero_when_no_inconclusive_items(three_item_data):
    """inconclusive_rate = 0 INCONCLUSIVE items / 3 = 0.0."""
    counts, results = three_item_data
    summary = compute_summary(counts, results, seed=42)
    assert summary.inconclusive_rate == pytest.approx(0.0, abs=1e-9)


def test_flaky_rate_nonzero_when_flaky_items_present():
    """flaky_rate = count(FLAKY) / n_items."""
    c1 = _counts("t1", 2, 5, 3, 5, True, True)
    c2 = _counts("t2", 5, 5, 5, 5, True, True)
    r1 = _result(c1, ItemClass.FLAKY)
    r2 = _result(c2, ItemClass.UNCHANGED_PASS)
    summary = compute_summary([c1, c2], [r1, r2], seed=42)
    assert summary.flaky_rate == pytest.approx(0.5, rel=1e-6)


def test_inconclusive_rate_nonzero_when_inconclusive_items_present():
    """inconclusive_rate = count(INCONCLUSIVE) / n_items."""
    c1 = _counts("t1", 2, 4, 3, 4, True, True)
    c2 = _counts("t2", 5, 5, 5, 5, True, True)
    r1 = _result(c1, ItemClass.INCONCLUSIVE)
    r2 = _result(c2, ItemClass.UNCHANGED_PASS)
    summary = compute_summary([c1, c2], [r1, r2], seed=42)
    assert summary.inconclusive_rate == pytest.approx(0.5, rel=1e-6)


def test_calls_used_is_sum_of_all_samples(three_item_data):
    """calls_used = sum of (base_n + cand_n) across all items.

    A: 4+4=8, B: 4+4=8, C: 4+4=8 -> total 24.
    """
    counts, results = three_item_data
    summary = compute_summary(counts, results, seed=42)
    assert summary.calls_used == 24


def test_calls_used_sums_correctly():
    """calls_used sums base_n + cand_n across items."""
    c1 = _counts("t1", 3, 5, 2, 6)
    c2 = _counts("t2", 1, 3, 1, 3)
    r1 = _result(c1, ItemClass.FLAKY)
    r2 = _result(c2, ItemClass.UNCHANGED_FAIL)
    summary = compute_summary([c1, c2], [r1, r2], seed=42)
    # (5+6) + (3+3) = 11 + 6 = 17
    assert summary.calls_used == 17


def test_mcnemar_p_large_when_no_difference():
    """When both sides always pass, McNemar p-value should be large (no difference)."""
    c1 = _counts("t1", 5, 5, 5, 5, True, True)
    c2 = _counts("t2", 5, 5, 5, 5, True, True)
    r1 = _result(c1, ItemClass.UNCHANGED_PASS)
    r2 = _result(c2, ItemClass.UNCHANGED_PASS)
    summary = compute_summary([c1, c2], [r1, r2], seed=42)
    # With no discordant pairs, McNemar p-value should be 1.0 (or None for degenerate case).
    if summary.mcnemar_p is not None:
        assert summary.mcnemar_p > 0.1


def test_n_items_zero_does_not_crash():
    """compute_summary with empty lists should return n_items=0 without crashing."""
    summary = compute_summary([], [], seed=42)
    assert summary.n_items == 0


def test_n_items_matches_input_count(three_item_data):
    """n_items in summary should equal the number of items passed in."""
    counts, results = three_item_data
    summary = compute_summary(counts, results, seed=42)
    assert summary.n_items == 3


def test_base_rate_is_mean_of_per_item_base_rates():
    """base_rate = mean(base_pass / base_n) over all items.

    Item A: 4/4 = 1.0, Item B: 0/4 = 0.0 -> mean = 0.5.
    """
    c_a = _counts("a", 4, 4, 4, 4)
    c_b = _counts("b", 0, 4, 4, 4)
    r_a = _result(c_a, ItemClass.UNCHANGED_PASS)
    r_b = _result(c_b, ItemClass.RELIABLE_IMPROVEMENT)
    summary = compute_summary([c_a, c_b], [r_a, r_b], seed=42)
    assert summary.base_rate == pytest.approx(0.5, rel=1e-6)


def test_cand_rate_is_mean_of_per_item_cand_rates():
    """cand_rate = mean(cand_pass / cand_n) over all items.

    Item A: cand 4/4 = 1.0, Item B: cand 0/4 = 0.0 -> mean = 0.5.
    """
    c_a = _counts("a", 4, 4, 4, 4)
    c_b = _counts("b", 4, 4, 0, 4)
    r_a = _result(c_a, ItemClass.UNCHANGED_PASS)
    r_b = _result(c_b, ItemClass.RELIABLE_REGRESSION)
    summary = compute_summary([c_a, c_b], [r_a, r_b], seed=42)
    assert summary.cand_rate == pytest.approx(0.5, rel=1e-6)
