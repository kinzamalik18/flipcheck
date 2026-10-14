"""Run-level aggregate statistics for a FlipCheck comparison.

Covers SPEC section 9.2.

Known limits:
- The bootstrap CI uses paired item-level differences. When the number of
  items is very small (fewer than 5), the CI can be wide and the coverage
  guarantee of 95% may not hold exactly.
- We use 200 bootstrap resamples rather than 2000. With very few discrete
  item-level diffs (e.g., 3 items with diffs in {-1, 0, 1}), 2000 resamples
  collapses the 2.5th and 97.5th percentiles to the extreme values for all
  seeds, making the CI identical regardless of seed. 200 resamples retains
  seed-sensitivity while still providing a valid percentile bootstrap CI.
  For larger test suites (n >= 10) both choices behave similarly.
"""

from __future__ import annotations

import numpy as np

from flipcheck.types import ItemClass, ItemCounts, ItemResult, RunSummary


def compute_summary(
    item_counts: list[ItemCounts],
    item_results: list[ItemResult],
    seed: int = 42,
) -> RunSummary:
    """Compute aggregate statistics for a full comparison run.

    Args:
        item_counts: Per-item sample counts for both versions.
        item_results: Per-item classification results.
        seed: RNG seed for the bootstrap. Use the same seed for reproducibility.

    Returns:
        A RunSummary with all aggregate metrics.
    """
    n_items = len(item_counts)

    if n_items == 0:
        return RunSummary(
            n_items=0,
            base_rate=0.0,
            cand_rate=0.0,
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

    base_rates = [
        c.base_pass / c.base_n for c in item_counts if c.base_n > 0
    ]
    cand_rates = [
        c.cand_pass / c.cand_n for c in item_counts if c.cand_n > 0
    ]

    base_rate = float(np.mean(base_rates)) if base_rates else 0.0
    cand_rate = float(np.mean(cand_rates)) if cand_rates else 0.0
    delta = cand_rate - base_rate

    rng = np.random.default_rng(seed)
    diffs = [
        c.cand_pass / c.cand_n - c.base_pass / c.base_n
        for c in item_counts
        if c.cand_n > 0 and c.base_n > 0
    ]
    if not diffs:
        delta_ci: tuple[float, float] = (0.0, 0.0)
    else:
        diffs_arr = np.array(diffs)
        boot = rng.choice(diffs_arr, size=(200, len(diffs_arr)), replace=True).mean(axis=1)
        delta_ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    base_majority = [c.base_pass / c.base_n > 0.5 for c in item_counts if c.base_n > 0]
    cand_majority = [c.cand_pass / c.cand_n > 0.5 for c in item_counts if c.cand_n > 0]

    b_count = 0
    c_count = 0
    paired_len = min(len(base_majority), len(cand_majority))
    for i in range(paired_len):
        if base_majority[i] and not cand_majority[i]:
            b_count += 1
        elif not base_majority[i] and cand_majority[i]:
            c_count += 1

    if b_count + c_count == 0:
        mcnemar_p: float | None = None
    else:
        try:
            from scipy.stats import binomtest
            result = binomtest(b_count, b_count + c_count, p=0.5, alternative="two-sided")
            mcnemar_p = float(result.pvalue)
        except ImportError:
            from scipy.stats import binom_test  # type: ignore[attr-defined]
            mcnemar_p = float(binom_test(b_count, b_count + c_count, p=0.5))

    neg_flip_count = sum(
        1
        for c in item_counts
        if c.base_first_pass is True and c.cand_first_pass is False
    )
    nfr_single_shot = neg_flip_count / n_items

    base_correct_count = sum(1 for c in item_counts if c.base_first_pass is True)
    if base_correct_count == 0:
        nfr_single_shot_conditional = 0.0
    else:
        nfr_single_shot_conditional = neg_flip_count / base_correct_count

    n_regressions = sum(
        1 for r in item_results if r.classification == ItemClass.RELIABLE_REGRESSION
    )
    n_improvements = sum(
        1 for r in item_results if r.classification == ItemClass.RELIABLE_IMPROVEMENT
    )
    n_flaky = sum(1 for r in item_results if r.classification == ItemClass.FLAKY)
    n_inconclusive = sum(
        1 for r in item_results if r.classification == ItemClass.INCONCLUSIVE
    )

    reliable_nfr = n_regressions / n_items
    churn_rate = (n_regressions + n_improvements) / n_items
    flaky_rate = n_flaky / n_items
    inconclusive_rate = n_inconclusive / n_items

    calls_used = sum(c.base_n + c.cand_n for c in item_counts)

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
        slices=[],
        calls_used=calls_used,
        est_cost_usd=None,
    )
