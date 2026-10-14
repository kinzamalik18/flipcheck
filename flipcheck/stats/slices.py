"""Tag-based slice summaries for a FlipCheck comparison run.

Covers SPEC section 9.3.
"""

from __future__ import annotations

from flipcheck.types import ItemClass, ItemCounts, ItemResult, SliceSummary, TestCase


def compute_slices(
    cases: list[TestCase],
    counts: list[ItemCounts],
    item_results: list[ItemResult],
) -> list[SliceSummary]:
    """Compute per-tag slice summaries from classified item results.

    For each unique tag found across all cases, builds a SliceSummary with
    aggregate pass rates and reliable change counts. Also adds a synthetic
    'critical' slice covering all cases marked critical=True.

    A slice with fewer than 10 items is flagged low_confidence=True, meaning
    the pass rate estimates are unreliable and should not trigger gate rules.

    Args:
        cases: Test cases with tag and critical metadata.
        counts: Per-item sample counts, aligned by case_id.
        item_results: Per-item classification results, aligned by case_id.

    Returns:
        List of SliceSummary, one per unique tag plus the 'critical' slice
        when at least one critical case exists.
    """
    counts_map: dict[str, ItemCounts] = {c.case_id: c for c in counts}
    results_map: dict[str, ItemResult] = {r.counts.case_id: r for r in item_results}

    tags_to_ids: dict[str, list[str]] = {}
    for case in cases:
        for tag in case.tags:
            tags_to_ids.setdefault(tag, []).append(case.id)

    critical_ids = [case.id for case in cases if case.critical]

    slices: list[SliceSummary] = []

    all_tag_groups = list(tags_to_ids.items())
    if critical_ids:
        all_tag_groups.append(("critical", critical_ids))

    for tag, case_ids in all_tag_groups:
        slice_counts = [counts_map[cid] for cid in case_ids if cid in counts_map]
        slice_results = [results_map[cid] for cid in case_ids if cid in results_map]

        if not slice_counts:
            continue

        base_rate = sum(
            c.base_pass / c.base_n for c in slice_counts if c.base_n > 0
        ) / len(slice_counts)
        cand_rate = sum(
            c.cand_pass / c.cand_n for c in slice_counts if c.cand_n > 0
        ) / len(slice_counts)
        delta = cand_rate - base_rate

        regressions = sum(
            1 for r in slice_results if r.classification == ItemClass.RELIABLE_REGRESSION
        )
        improvements = sum(
            1 for r in slice_results if r.classification == ItemClass.RELIABLE_IMPROVEMENT
        )

        slices.append(
            SliceSummary(
                tag=tag,
                n_items=len(slice_counts),
                base_rate=base_rate,
                cand_rate=cand_rate,
                delta=delta,
                regressions=regressions,
                improvements=improvements,
                low_confidence=len(slice_counts) < 10,
            )
        )

    return slices
