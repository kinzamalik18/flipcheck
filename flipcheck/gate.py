"""Verdict logic for FlipCheck.

Computes the SHIP, SHIP_WITH_CAVEATS or BLOCK verdict from run results.
Covers SPEC section 10.
"""

from __future__ import annotations

from flipcheck.types import (
    GateConfig,
    ItemClass,
    ItemResult,
    Reason,
    RunSummary,
    Verdict,
    VerdictLevel,
)


def decide(
    summary: RunSummary,
    items: list[ItemResult],
    critical: set[str],
    gates: GateConfig,
) -> Verdict:
    """Compute the ship/block verdict from run results per SPEC section 10.

    All BLOCK conditions are evaluated first, then SHIP_WITH_CAVEATS conditions.
    The level is BLOCK if any BLOCK reason fired, SHIP_WITH_CAVEATS if any
    caveat reason fired but no BLOCK reason, and SHIP otherwise.

    Reasons are sorted with BLOCK reasons before SHIP_WITH_CAVEATS reasons,
    then by descending number of case IDs within each level group.

    Args:
        summary: Aggregate statistics for the run.
        items: Per-item classification results.
        critical: Set of case IDs marked critical.
        gates: Thresholds from the project config.

    Returns:
        A Verdict with a level and the list of reasons that contributed.
    """
    block_reasons: list[Reason] = []
    caveat_reasons: list[Reason] = []

    regressions = [r for r in items if r.classification == ItemClass.RELIABLE_REGRESSION]
    critical_regressions = [r for r in regressions if r.counts.case_id in critical]
    non_critical_regressions = [r for r in regressions if r.counts.case_id not in critical]

    if critical_regressions and gates.block_on_critical_regression:
        block_reasons.append(
            Reason(
                level=VerdictLevel.BLOCK,
                message=(
                    f"{len(critical_regressions)} critical case(s) regressed reliably."
                ),
                case_ids=[r.counts.case_id for r in critical_regressions],
            )
        )

    if summary.reliable_nfr > gates.max_reliable_nfr:
        block_reasons.append(
            Reason(
                level=VerdictLevel.BLOCK,
                message=(
                    f"Reliable negative flip rate {summary.reliable_nfr:.1%} "
                    f"exceeds limit {gates.max_reliable_nfr:.1%}."
                ),
            )
        )

    if summary.delta_ci[1] < 0.0:
        block_reasons.append(
            Reason(
                level=VerdictLevel.BLOCK,
                message=(
                    "Overall quality dropped: the upper bound of the delta CI "
                    f"({summary.delta_ci[1]:.3f}) is below zero."
                ),
            )
        )

    if non_critical_regressions:
        caveat_reasons.append(
            Reason(
                level=VerdictLevel.SHIP_WITH_CAVEATS,
                message=(
                    f"{len(non_critical_regressions)} non-critical case(s) "
                    "regressed reliably."
                ),
                case_ids=[r.counts.case_id for r in non_critical_regressions],
            )
        )

    if critical_regressions and not gates.block_on_critical_regression:
        caveat_reasons.append(
            Reason(
                level=VerdictLevel.SHIP_WITH_CAVEATS,
                message=(
                    f"{len(critical_regressions)} critical case(s) regressed reliably "
                    "(block_on_critical_regression is disabled)."
                ),
                case_ids=[r.counts.case_id for r in critical_regressions],
            )
        )

    if summary.flaky_rate > gates.max_flaky_rate:
        caveat_reasons.append(
            Reason(
                level=VerdictLevel.SHIP_WITH_CAVEATS,
                message=(
                    f"Flaky rate {summary.flaky_rate:.1%} exceeds limit "
                    f"{gates.max_flaky_rate:.1%}."
                ),
            )
        )

    if summary.inconclusive_rate > gates.max_inconclusive_rate:
        caveat_reasons.append(
            Reason(
                level=VerdictLevel.SHIP_WITH_CAVEATS,
                message=(
                    f"Inconclusive rate {summary.inconclusive_rate:.1%} exceeds limit "
                    f"{gates.max_inconclusive_rate:.1%}."
                ),
            )
        )

    for sl in summary.slices:
        if not sl.low_confidence and sl.delta < -gates.max_slice_drop:
            caveat_reasons.append(
                Reason(
                    level=VerdictLevel.SHIP_WITH_CAVEATS,
                    message=(
                        f"Slice '{sl.tag}' dropped by {-sl.delta:.1%}, "
                        f"exceeding the limit of {gates.max_slice_drop:.1%}."
                    ),
                )
            )

    if block_reasons:
        level = VerdictLevel.BLOCK
    elif caveat_reasons:
        level = VerdictLevel.SHIP_WITH_CAVEATS
    else:
        level = VerdictLevel.SHIP

    all_reasons = block_reasons + caveat_reasons
    all_reasons.sort(
        key=lambda r: (0 if r.level == VerdictLevel.BLOCK else 1, -len(r.case_ids))
    )

    return Verdict(level=level, reasons=all_reasons)
