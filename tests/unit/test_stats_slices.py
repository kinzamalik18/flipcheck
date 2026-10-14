"""Tests for flipcheck/stats/slices.py - tag-based slice summaries.

All tests are derived from SPEC section 9.3. The implementation is a stub
at the time of writing; these tests define the required behavior.
"""

from __future__ import annotations

import pytest

from flipcheck.types import ItemClass, ItemCounts, ItemResult, TestCase

# Skip the whole module gracefully when the implementation is not yet built.
try:
    import flipcheck.stats.slices as _slices_mod

    compute_slices = _slices_mod.compute_slices
except AttributeError as _e:
    pytest.skip(f"flipcheck.stats.slices not yet implemented: {_e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _case(case_id: str, tags: list[str] | None = None, critical: bool = False) -> TestCase:
    return TestCase(
        id=case_id,
        input={"q": case_id},
        tags=tags if tags is not None else [],
        critical=critical,
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


def _result(counts: ItemCounts, classification: ItemClass) -> ItemResult:
    return ItemResult(
        counts=counts,
        classification=classification,
        p_regression=None,
        p_improvement=None,
    )


def _slice_for_tag(slices, tag: str):
    """Return the SliceSummary with the given tag, or None."""
    for s in slices:
        if s.tag == tag:
            return s
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_items_with_tag_appear_in_correct_slice():
    """Items tagged 'urdu' should appear in a slice with tag='urdu'."""
    cases = [_case("t1", tags=["urdu"]), _case("t2", tags=["urdu"])]
    counts = [_counts("t1", 4, 5, 4, 5), _counts("t2", 3, 5, 3, 5)]
    results = [
        _result(counts[0], ItemClass.UNCHANGED_PASS),
        _result(counts[1], ItemClass.UNCHANGED_PASS),
    ]
    slices = compute_slices(cases, counts, results)
    urdu_slice = _slice_for_tag(slices, "urdu")
    assert urdu_slice is not None
    assert urdu_slice.n_items == 2


def test_items_without_tags_do_not_appear_in_tag_slices():
    """Items with no tags should not show up in any named tag slice."""
    cases = [_case("t1", tags=[]), _case("t2", tags=[])]
    counts = [_counts("t1", 4, 5, 4, 5), _counts("t2", 3, 5, 3, 5)]
    results = [
        _result(counts[0], ItemClass.UNCHANGED_PASS),
        _result(counts[1], ItemClass.UNCHANGED_PASS),
    ]
    slices = compute_slices(cases, counts, results)
    # Since no cases are critical and no tags exist, no tag slices should appear
    # (critical slice either absent or empty).
    non_critical_slices = [s for s in slices if s.tag != "critical"]
    assert non_critical_slices == []


def test_slice_with_fewer_than_10_items_has_low_confidence_true():
    """A slice with n_items < 10 should have low_confidence=True."""
    cases = [_case(f"t{i}", tags=["rare"]) for i in range(9)]
    counts = [_counts(f"t{i}", 4, 5, 4, 5) for i in range(9)]
    results = [_result(c, ItemClass.UNCHANGED_PASS) for c in counts]
    slices = compute_slices(cases, counts, results)
    rare_slice = _slice_for_tag(slices, "rare")
    assert rare_slice is not None
    assert rare_slice.low_confidence is True


def test_slice_with_10_or_more_items_has_low_confidence_false():
    """A slice with n_items >= 10 should have low_confidence=False."""
    cases = [_case(f"t{i}", tags=["common"]) for i in range(10)]
    counts = [_counts(f"t{i}", 4, 5, 4, 5) for i in range(10)]
    results = [_result(c, ItemClass.UNCHANGED_PASS) for c in counts]
    slices = compute_slices(cases, counts, results)
    common_slice = _slice_for_tag(slices, "common")
    assert common_slice is not None
    assert common_slice.low_confidence is False


def test_critical_synthetic_slice_covers_all_critical_cases():
    """The 'critical' slice should contain all cases marked critical=True."""
    cases = [
        _case("t1", tags=[], critical=True),
        _case("t2", tags=[], critical=True),
        _case("t3", tags=[], critical=False),
    ]
    counts = [
        _counts("t1", 5, 5, 5, 5),
        _counts("t2", 4, 5, 4, 5),
        _counts("t3", 3, 5, 3, 5),
    ]
    results = [
        _result(counts[0], ItemClass.UNCHANGED_PASS),
        _result(counts[1], ItemClass.UNCHANGED_PASS),
        _result(counts[2], ItemClass.UNCHANGED_PASS),
    ]
    slices = compute_slices(cases, counts, results)
    critical_slice = _slice_for_tag(slices, "critical")
    assert critical_slice is not None
    assert critical_slice.n_items == 2


def test_slice_base_rate_is_mean_of_per_item_base_rates():
    """base_rate in a slice is the mean of per-item base pass rates.

    t1: base 4/5 = 0.8, t2: base 2/5 = 0.4 -> mean = 0.6.
    """
    cases = [_case("t1", tags=["grp"]), _case("t2", tags=["grp"])]
    counts = [_counts("t1", 4, 5, 4, 5), _counts("t2", 2, 5, 4, 5)]
    results = [
        _result(counts[0], ItemClass.UNCHANGED_PASS),
        _result(counts[1], ItemClass.UNCHANGED_PASS),
    ]
    slices = compute_slices(cases, counts, results)
    grp_slice = _slice_for_tag(slices, "grp")
    assert grp_slice is not None
    assert grp_slice.base_rate == pytest.approx(0.6, rel=1e-6)


def test_slice_cand_rate_is_mean_of_per_item_cand_rates():
    """cand_rate in a slice is the mean of per-item candidate pass rates.

    t1: cand 5/5 = 1.0, t2: cand 1/5 = 0.2 -> mean = 0.6.
    """
    cases = [_case("t1", tags=["grp"]), _case("t2", tags=["grp"])]
    counts = [_counts("t1", 3, 5, 5, 5), _counts("t2", 3, 5, 1, 5)]
    results = [
        _result(counts[0], ItemClass.UNCHANGED_PASS),
        _result(counts[1], ItemClass.UNCHANGED_PASS),
    ]
    slices = compute_slices(cases, counts, results)
    grp_slice = _slice_for_tag(slices, "grp")
    assert grp_slice is not None
    assert grp_slice.cand_rate == pytest.approx(0.6, rel=1e-6)


def test_slice_delta_equals_cand_rate_minus_base_rate():
    """delta in a slice should equal cand_rate - base_rate.

    t1: base 0.8, cand 1.0; t2: base 0.4, cand 0.4.
    slice base_rate = mean(0.8, 0.4) = 0.6
    slice cand_rate = mean(1.0, 0.4) = 0.7
    delta = 0.7 - 0.6 = 0.1
    """
    cases = [_case("t1", tags=["grp"]), _case("t2", tags=["grp"])]
    counts = [_counts("t1", 4, 5, 5, 5), _counts("t2", 2, 5, 2, 5)]
    results = [
        _result(counts[0], ItemClass.RELIABLE_IMPROVEMENT),
        _result(counts[1], ItemClass.UNCHANGED_PASS),
    ]
    slices = compute_slices(cases, counts, results)
    grp_slice = _slice_for_tag(slices, "grp")
    assert grp_slice is not None
    assert grp_slice.delta == pytest.approx(
        grp_slice.cand_rate - grp_slice.base_rate, abs=1e-9
    )


def test_slice_regression_count_matches_reliable_regression_items():
    """regressions in a slice should count RELIABLE_REGRESSION items in that tag."""
    cases = [
        _case("t1", tags=["lang"]),
        _case("t2", tags=["lang"]),
        _case("t3", tags=["lang"]),
    ]
    counts = [
        _counts("t1", 5, 5, 0, 5),
        _counts("t2", 5, 5, 0, 5),
        _counts("t3", 5, 5, 5, 5),
    ]
    results = [
        _result(counts[0], ItemClass.RELIABLE_REGRESSION),
        _result(counts[1], ItemClass.RELIABLE_REGRESSION),
        _result(counts[2], ItemClass.UNCHANGED_PASS),
    ]
    slices = compute_slices(cases, counts, results)
    lang_slice = _slice_for_tag(slices, "lang")
    assert lang_slice is not None
    assert lang_slice.regressions == 2


def test_slice_improvement_count_matches_reliable_improvement_items():
    """improvements in a slice should count RELIABLE_IMPROVEMENT items in that tag."""
    cases = [
        _case("t1", tags=["domain"]),
        _case("t2", tags=["domain"]),
    ]
    counts = [
        _counts("t1", 0, 5, 5, 5),
        _counts("t2", 5, 5, 5, 5),
    ]
    results = [
        _result(counts[0], ItemClass.RELIABLE_IMPROVEMENT),
        _result(counts[1], ItemClass.UNCHANGED_PASS),
    ]
    slices = compute_slices(cases, counts, results)
    domain_slice = _slice_for_tag(slices, "domain")
    assert domain_slice is not None
    assert domain_slice.improvements == 1


def test_item_in_multiple_tags_appears_in_each_slice():
    """An item tagged with two tags should be counted in both tag slices."""
    cases = [_case("t1", tags=["a", "b"])]
    counts = [_counts("t1", 4, 5, 4, 5)]
    results = [_result(counts[0], ItemClass.UNCHANGED_PASS)]
    slices = compute_slices(cases, counts, results)
    slice_a = _slice_for_tag(slices, "a")
    slice_b = _slice_for_tag(slices, "b")
    assert slice_a is not None
    assert slice_b is not None
    assert slice_a.n_items == 1
    assert slice_b.n_items == 1


def test_critical_slice_absent_or_empty_when_no_critical_cases():
    """The 'critical' slice should not appear when no cases are critical, or it should have n_items=0."""
    cases = [_case("t1", tags=[], critical=False)]
    counts = [_counts("t1", 4, 5, 4, 5)]
    results = [_result(counts[0], ItemClass.UNCHANGED_PASS)]
    slices = compute_slices(cases, counts, results)
    critical_slice = _slice_for_tag(slices, "critical")
    # Either the slice is absent or it has n_items=0.
    if critical_slice is not None:
        assert critical_slice.n_items == 0
