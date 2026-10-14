"""Adaptive sampling policy for FlipCheck.

Covers SPEC section 8.2.

The policy is a pure function: given current sample counts, the sampling
config, the remaining call budget, and which cases are critical, it returns
a dict of {case_id: extra_samples_per_version} for the next round.

An empty dict means all items are either settled or at k_max, or the budget
is exhausted.

Known limit: repeated looks at the data (each adaptive round re-classifies
items) inflate the false positive rate compared to a single fixed-k design.
The simulation in tests/simulation/ measures the actual false alarm rate
under adaptive sampling. The SPEC notes this and suggests tuning alpha if
inflation is unacceptable in practice.
"""

from __future__ import annotations

from flipcheck.types import ItemCounts, SamplingConfig


def _is_settled(c: ItemCounts) -> bool:
    """Return True when both versions are unanimous (all-pass or all-fail).

    A unanimous result on both sides is a strong enough signal to stop
    sampling without waiting for a statistically significant Fisher p-value.
    This is a simpler proxy than running classify_items; it avoids circular
    dependency between the sampling policy and the stats module.

    An item with base_n=0 or cand_n=0 has no data yet and is not settled.
    """
    if c.base_n == 0 or c.cand_n == 0:
        return False

    base_all_pass = c.base_pass == c.base_n
    base_all_fail = c.base_pass == 0
    cand_all_pass = c.cand_pass == c.cand_n
    cand_all_fail = c.cand_pass == 0

    both_all_pass = base_all_pass and cand_all_pass
    both_all_fail = base_all_fail and cand_all_fail
    return both_all_pass or both_all_fail


def next_round(
    counts: list[ItemCounts],
    cfg: SamplingConfig,
    budget_left: int,
    critical: set[str],
) -> dict[str, int]:
    """Return the set of items to sample in the next adaptive round.

    Filters to unsettled items below k_max, then prioritizes by:
    1. Critical items first.
    2. Largest absolute pass rate difference between versions.
    3. Alphabetical case_id as a tie-breaker.

    Each scheduled item gets cfg.step extra samples per version (so
    cfg.step * 2 total calls per item).

    Args:
        counts: Current sample counts for all items.
        cfg: Sampling configuration (k_max, step).
        budget_left: Remaining call budget (total calls, both versions).
        critical: Set of case IDs marked critical.

    Returns:
        Mapping from case_id to extra samples per version. Empty when done.
    """
    if budget_left <= 0:
        return {}

    candidates: list[ItemCounts] = [
        c for c in counts if not _is_settled(c) and c.base_n < cfg.k_max
    ]

    if not candidates:
        return {}

    def sort_key(c: ItemCounts) -> tuple[int, float, str]:
        is_critical = 0 if c.case_id in critical else 1
        base_rate = c.base_pass / c.base_n if c.base_n > 0 else 0.0
        cand_rate = c.cand_pass / c.cand_n if c.cand_n > 0 else 0.0
        rate_diff = -abs(cand_rate - base_rate)
        return (is_critical, rate_diff, c.case_id)

    candidates.sort(key=sort_key)

    cost_per_item = cfg.step * 2
    scheduled: dict[str, int] = {}
    remaining = budget_left

    for c in candidates:
        if remaining < cost_per_item:
            break
        scheduled[c.case_id] = cfg.step
        remaining -= cost_per_item

    return scheduled
