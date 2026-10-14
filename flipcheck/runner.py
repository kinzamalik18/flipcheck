"""Runner for FlipCheck: orchestrates fixed-mode comparison runs.

Covers SPEC section 8.1 (fixed mode).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from flipcheck import gate
from flipcheck.config import compute_version_id
from flipcheck.providers import get_provider
from flipcheck.scorers import build_scorers
from flipcheck.store.db import SQLiteStore
from flipcheck.types import (
    ItemClass,
    ItemCounts,
    ItemResult,
    ProjectConfig,
    RunSummary,
    Sample,
    TestCase,
    Verdict,
    Version,
)


async def run_fixed(
    config: ProjectConfig,
    store: SQLiteStore,
    suite: list[TestCase],
    run_id: str | None = None,
) -> tuple[str, RunSummary, Verdict]:
    """Run a fixed-mode comparison.

    Collects k samples per case for each version (baseline and candidate),
    scores them, builds ItemCounts, classifies items, and returns a verdict.

    Args:
        config: Project configuration including targets, sampling settings and gates.
        store: Persistent store for samples and results.
        suite: List of test cases to run.
        run_id: Optional run identifier. A new UUID is generated when not provided.

    Returns:
        A 3-tuple of (run_id, RunSummary, Verdict).
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    k = config.sampling.k
    budget_calls = config.sampling.budget_calls
    calls_used = 0

    version_names = ["baseline", "candidate"]
    versions: dict[str, Version] = {}
    for name in version_names:
        target_cfg = config.targets[name]
        vid = compute_version_id(target_cfg, None)
        versions[name] = Version(name=name, config=target_cfg, version_id=vid)

    base_version = versions["baseline"]
    cand_version = versions["candidate"]

    base_pass_by_case: dict[str, int] = {}
    cand_pass_by_case: dict[str, int] = {}
    base_n_by_case: dict[str, int] = {}
    cand_n_by_case: dict[str, int] = {}
    base_first_pass: dict[str, bool | None] = {}
    cand_first_pass: dict[str, bool | None] = {}

    for case in suite:
        base_pass_by_case[case.id] = 0
        cand_pass_by_case[case.id] = 0
        base_n_by_case[case.id] = 0
        cand_n_by_case[case.id] = 0
        base_first_pass[case.id] = None
        cand_first_pass[case.id] = None

    for case in suite:
        scorers = build_scorers(case, config.default_scorers)

        for version, pass_map, n_map, first_pass_map in [
            (base_version, base_pass_by_case, base_n_by_case, base_first_pass),
            (cand_version, cand_pass_by_case, cand_n_by_case, cand_first_pass),
        ]:
            provider = get_provider(version)

            existing = {
                s.sample_idx: s
                for s in store.get_samples(run_id, case.id, version.version_id)
            }

            for sample_idx in range(k):
                if calls_used >= budget_calls:
                    break

                if sample_idx in existing:
                    sample = existing[sample_idx].model_copy(update={"cached": True})
                    store.save_sample(run_id, sample)
                else:
                    sample = await provider.generate(version, case, sample_idx)
                    store.save_sample(run_id, sample)
                    calls_used += 1

                passed = await _score_sample(sample, scorers, case)
                n_map[case.id] += 1
                if passed:
                    pass_map[case.id] += 1
                if sample_idx == 0:
                    first_pass_map[case.id] = passed

    item_results: list[ItemResult] = []
    for case in suite:
        counts = ItemCounts(
            case_id=case.id,
            base_pass=base_pass_by_case[case.id],
            base_n=base_n_by_case[case.id],
            cand_pass=cand_pass_by_case[case.id],
            cand_n=cand_n_by_case[case.id],
            base_first_pass=base_first_pass[case.id],
            cand_first_pass=cand_first_pass[case.id],
        )
        classification = _classify(counts)
        item_results.append(ItemResult(counts=counts, classification=classification))

    store.save_item_results(run_id, item_results)

    summary = _build_summary(suite, item_results, calls_used)

    critical = {c.id for c in suite if c.critical}
    verdict = gate.decide(summary, item_results, critical, config.gates)

    finished_at = datetime.now(tz=UTC).isoformat()
    store.update_run_finished(
        run_id=run_id,
        status="done",
        finished_at=finished_at,
        calls_used=calls_used,
        est_cost_usd=None,
    )

    return run_id, summary, verdict


async def _score_sample(
    sample: Sample,
    scorers: list,
    case: TestCase,
) -> bool:
    """Return True when the sample output passes all scorers.

    Returns False immediately when the sample has an error field set.
    A sample with no scorers configured always passes.
    """
    if sample.error:
        return False

    if not scorers:
        return True

    output = sample.output or ""
    results = [await s.score(case, output) for s in scorers]
    return all(r.passed for r in results)


def _classify(counts: ItemCounts) -> ItemClass:
    """Classify one item using a simplified heuristic (Phase 2 placeholder).

    Phase 3 will replace this with Fisher exact tests and BH adjustment.
    """
    base_n = counts.base_n
    cand_n = counts.cand_n

    if base_n == 0 or cand_n == 0:
        return ItemClass.INCONCLUSIVE

    base_rate = counts.base_pass / base_n
    cand_rate = counts.cand_pass / cand_n

    if base_rate > 0.5 and cand_rate < 0.5:
        return ItemClass.RELIABLE_REGRESSION
    if cand_rate > 0.5 and base_rate < 0.5:
        return ItemClass.RELIABLE_IMPROVEMENT
    if base_rate >= 0.5 and cand_rate >= 0.5:
        return ItemClass.UNCHANGED_PASS
    return ItemClass.UNCHANGED_FAIL


def _build_summary(
    suite: list[TestCase],
    item_results: list[ItemResult],
    calls_used: int,
) -> RunSummary:
    """Build a RunSummary from item results.

    Statistics are approximate placeholders until the Phase 3 stats module
    replaces them with Fisher exact tests and bootstrap confidence intervals.
    """
    n_items = len(suite)

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
            calls_used=calls_used,
            est_cost_usd=None,
        )

    base_rates = [
        r.counts.base_pass / r.counts.base_n
        for r in item_results
        if r.counts.base_n > 0
    ]
    cand_rates = [
        r.counts.cand_pass / r.counts.cand_n
        for r in item_results
        if r.counts.cand_n > 0
    ]

    base_rate = sum(base_rates) / n_items if base_rates else 0.0
    cand_rate = sum(cand_rates) / n_items if cand_rates else 0.0
    delta = cand_rate - base_rate

    reliable_regressions = [
        r
        for r in item_results
        if r.classification == ItemClass.RELIABLE_REGRESSION
    ]
    reliable_improvements = [
        r
        for r in item_results
        if r.classification == ItemClass.RELIABLE_IMPROVEMENT
    ]
    reliable_nfr = len(reliable_regressions) / max(n_items, 1)
    churn_rate = (len(reliable_regressions) + len(reliable_improvements)) / max(
        n_items, 1
    )

    return RunSummary(
        n_items=n_items,
        base_rate=base_rate,
        cand_rate=cand_rate,
        delta=delta,
        delta_ci=(delta - 0.1, delta + 0.1),
        mcnemar_p=None,
        nfr_single_shot=0.0,
        nfr_single_shot_conditional=0.0,
        reliable_nfr=reliable_nfr,
        churn_rate=churn_rate,
        flaky_rate=0.0,
        inconclusive_rate=0.0,
        slices=[],
        calls_used=calls_used,
        est_cost_usd=None,
    )
