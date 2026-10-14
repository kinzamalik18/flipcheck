"""Verdict logic for FlipCheck.

Computes the SHIP, SHIP_WITH_CAVEATS or BLOCK verdict from run results.
Covers SPEC section 8 (gate).
"""

from __future__ import annotations

from flipcheck.types import (
    GateConfig,
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
    """Compute the ship/block verdict from run results.

    Checks are applied in priority order: BLOCK conditions first, then
    SHIP_WITH_CAVEATS conditions. The first BLOCK condition found sets the
    level; additional conditions append extra Reason objects.

    Args:
        summary: Aggregate statistics for the run.
        items: Per-item classification results.
        critical: Set of case IDs marked critical.
        gates: Thresholds from the project config.

    Returns:
        A Verdict with a level and the list of reasons that contributed.
    """
    reasons: list[Reason] = []
    level = VerdictLevel.SHIP

    regressions = [r for r in items if r.classification.value == "reliable_regression"]
    critical_regressions = [r for r in regressions if r.counts.case_id in critical]
    non_critical_regressions = [
        r for r in regressions if r.counts.case_id not in critical
    ]

    if critical_regressions and gates.block_on_critical_regression:
        level = VerdictLevel.BLOCK
        reasons.append(
            Reason(
                level=VerdictLevel.BLOCK,
                message=(
                    f"{len(critical_regressions)} critical case(s) regressed reliably."
                ),
                case_ids=[r.counts.case_id for r in critical_regressions],
            )
        )

    if summary.reliable_nfr > gates.max_reliable_nfr:
        level = VerdictLevel.BLOCK
        reasons.append(
            Reason(
                level=VerdictLevel.BLOCK,
                message=(
                    f"Reliable negative flip rate {summary.reliable_nfr:.1%} "
                    f"exceeds limit {gates.max_reliable_nfr:.1%}."
                ),
            )
        )

    if non_critical_regressions and level != VerdictLevel.BLOCK:
        level = VerdictLevel.SHIP_WITH_CAVEATS
        reasons.append(
            Reason(
                level=VerdictLevel.SHIP_WITH_CAVEATS,
                message=(
                    f"{len(non_critical_regressions)} non-critical case(s) "
                    "regressed reliably."
                ),
                case_ids=[r.counts.case_id for r in non_critical_regressions],
            )
        )

    return Verdict(level=level, reasons=reasons)
