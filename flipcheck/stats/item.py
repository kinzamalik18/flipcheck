"""Per-item classification using Fisher exact tests and BH adjustment.

Covers SPEC section 9.1.

Known limit: with only 5 samples per side the smallest achievable Fisher
p-value is about 0.004. When many items are tested, Benjamini-Hochberg
inflates the threshold so the effective per-item alpha is smaller. The
adaptive sampler compensates by collecting up to k_max samples for items
that remain uncertain.
"""

from __future__ import annotations

from scipy.stats import fisher_exact

from flipcheck.types import ItemClass, ItemCounts, ItemResult


def fisher_p_regression(a: int, n1: int, b: int, n2: int) -> float:
    """One-sided Fisher exact p-value for the hypothesis that the candidate regressed.

    Tests whether the baseline pass rate (a/n1) is significantly higher than
    the candidate pass rate (b/n2). A small p-value means the candidate is
    reliably worse.

    The contingency table is [[a, n1-a], [b, n2-b]] where row 0 is baseline
    and row 1 is candidate. alternative='greater' tests whether the odds ratio
    (baseline odds) / (candidate odds) is greater than 1, i.e. baseline is
    better, i.e. candidate regressed.

    Args:
        a: Baseline pass count.
        n1: Baseline total samples.
        b: Candidate pass count.
        n2: Candidate total samples.

    Returns:
        p-value from scipy.stats.fisher_exact with alternative='greater'.
    """
    table = [[a, n1 - a], [b, n2 - b]]
    _, p = fisher_exact(table, alternative="greater")
    return float(p)


def fisher_p_improvement(a: int, n1: int, b: int, n2: int) -> float:
    """One-sided Fisher exact p-value for the hypothesis that the candidate improved.

    Tests whether the candidate pass rate (b/n2) is significantly higher than
    the baseline pass rate (a/n1). A small p-value means the candidate is
    reliably better.

    Same table as fisher_p_regression but alternative='less' tests whether
    the baseline odds are lower, i.e. the candidate improved.

    Args:
        a: Baseline pass count.
        n1: Baseline total samples.
        b: Candidate pass count.
        n2: Candidate total samples.

    Returns:
        p-value from scipy.stats.fisher_exact with alternative='less'.
    """
    table = [[a, n1 - a], [b, n2 - b]]
    _, p = fisher_exact(table, alternative="less")
    return float(p)


def bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg false discovery rate adjustment.

    Adjusts a list of p-values to control the false discovery rate. Returns
    adjusted p-values in the same positional order as the input.

    Algorithm:
    1. Sort p-values ascending, retaining original indices.
    2. For each rank i (1-indexed from smallest), compute p * n / i.
    3. Apply a cumulative minimum from the largest rank down (step-up procedure).
    4. Clip to [0, 1].

    Args:
        p_values: Raw p-values, one per hypothesis.

    Returns:
        BH-adjusted p-values in the same order as p_values.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [0.0] * n
    for rank_zero, (orig_idx, p) in enumerate(indexed):
        rank = rank_zero + 1
        adjusted[orig_idx] = p * n / rank

    running_min = 1.0
    for rank_zero in range(n - 1, -1, -1):
        orig_idx = indexed[rank_zero][0]
        running_min = min(running_min, adjusted[orig_idx])
        adjusted[orig_idx] = running_min

    return [min(1.0, max(0.0, v)) for v in adjusted]


def classify_items(
    counts: list[ItemCounts],
    alpha: float,
    inconclusive_ids: set[str],
) -> list[ItemResult]:
    """Classify each test case after multiple sampling rounds.

    Applies Fisher exact tests (one per direction), BH correction across all
    items separately for regression and improvement p-values, then assigns
    each item a class per the following priority order:

    1. case_id in inconclusive_ids -> INCONCLUSIVE (budget ran out)
    2. adjusted p_regression < alpha -> RELIABLE_REGRESSION
    3. adjusted p_improvement < alpha -> RELIABLE_IMPROVEMENT
    4. either pass rate in [0.2, 0.8] -> FLAKY
    5. both pass rates >= 0.5 -> UNCHANGED_PASS
    6. otherwise -> UNCHANGED_FAIL

    INCONCLUSIVE is checked first so that items where the budget ran out
    before reaching k_max are not accidentally promoted to a reliable verdict
    by the Fisher test, even if the counts happen to be extreme.

    Pass rate for a side with n=0 is treated as 0.0.

    Args:
        counts: Per-item sample counts for both versions.
        alpha: Significance level (after BH adjustment).
        inconclusive_ids: Case IDs where the budget ran out before k_max.

    Returns:
        One ItemResult per input ItemCounts, in the same order.
    """
    reg_pvals = [
        fisher_p_regression(c.base_pass, c.base_n, c.cand_pass, c.cand_n)
        if c.base_n > 0 and c.cand_n > 0
        else 1.0
        for c in counts
    ]
    imp_pvals = [
        fisher_p_improvement(c.base_pass, c.base_n, c.cand_pass, c.cand_n)
        if c.base_n > 0 and c.cand_n > 0
        else 1.0
        for c in counts
    ]

    adj_reg = bh_adjust(reg_pvals)
    adj_imp = bh_adjust(imp_pvals)

    results: list[ItemResult] = []
    for i, c in enumerate(counts):
        base_rate = c.base_pass / c.base_n if c.base_n > 0 else 0.0
        cand_rate = c.cand_pass / c.cand_n if c.cand_n > 0 else 0.0

        if c.case_id in inconclusive_ids:
            cls = ItemClass.INCONCLUSIVE
        elif adj_reg[i] < alpha:
            cls = ItemClass.RELIABLE_REGRESSION
        elif adj_imp[i] < alpha:
            cls = ItemClass.RELIABLE_IMPROVEMENT
        elif 0.2 <= base_rate <= 0.8 or 0.2 <= cand_rate <= 0.8:
            cls = ItemClass.FLAKY
        elif base_rate >= 0.5 and cand_rate >= 0.5:
            cls = ItemClass.UNCHANGED_PASS
        else:
            cls = ItemClass.UNCHANGED_FAIL

        results.append(
            ItemResult(
                counts=c,
                classification=cls,
                p_regression=adj_reg[i],
                p_improvement=adj_imp[i],
            )
        )

    return results
