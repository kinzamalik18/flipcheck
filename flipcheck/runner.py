"""Runner for FlipCheck: orchestrates fixed-mode and adaptive-mode comparison runs.

Covers SPEC sections 8.1 (fixed mode) and 8.2 (adaptive mode).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from flipcheck import gate
from flipcheck.config import compute_version_id
from flipcheck.providers import get_provider
from flipcheck.sampling.adaptive import next_round
from flipcheck.scorers import build_scorers
from flipcheck.stats.aggregate import compute_summary
from flipcheck.stats.item import classify_items
from flipcheck.stats.slices import compute_slices
from flipcheck.store.db import SQLiteStore
from flipcheck.types import (
    ItemCounts,
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
    scores them, builds ItemCounts, classifies items, computes real aggregate
    statistics, and returns a verdict.

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

    item_counts = [
        ItemCounts(
            case_id=case.id,
            base_pass=base_pass_by_case[case.id],
            base_n=base_n_by_case[case.id],
            cand_pass=cand_pass_by_case[case.id],
            cand_n=cand_n_by_case[case.id],
            base_first_pass=base_first_pass[case.id],
            cand_first_pass=cand_first_pass[case.id],
        )
        for case in suite
    ]

    item_results = classify_items(item_counts, alpha=config.gates.alpha, inconclusive_ids=set())

    store.save_item_results(run_id, item_results)

    summary = compute_summary(item_counts, item_results, seed=42)
    summary = summary.model_copy(
        update={"slices": compute_slices(suite, item_counts, item_results)}
    )

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


async def run_adaptive(
    config: ProjectConfig,
    store: SQLiteStore,
    suite: list[TestCase],
    run_id: str | None = None,
) -> tuple[str, RunSummary, Verdict]:
    """Run an adaptive-mode comparison.

    Collects k_min samples per case first, then loops calling next_round to
    gather more samples for unsettled items until all items are settled,
    k_max is reached, or the budget is exhausted. Items that did not reach
    k_max when the budget ran out are marked INCONCLUSIVE.

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

    cfg = config.sampling
    budget_calls = cfg.budget_calls
    calls_used = 0

    version_names = ["baseline", "candidate"]
    versions: dict[str, Version] = {}
    for name in version_names:
        target_cfg = config.targets[name]
        vid = compute_version_id(target_cfg, None)
        versions[name] = Version(name=name, config=target_cfg, version_id=vid)

    base_version = versions["baseline"]
    cand_version = versions["candidate"]

    base_pass_by_case: dict[str, int] = {case.id: 0 for case in suite}
    cand_pass_by_case: dict[str, int] = {case.id: 0 for case in suite}
    base_n_by_case: dict[str, int] = {case.id: 0 for case in suite}
    cand_n_by_case: dict[str, int] = {case.id: 0 for case in suite}
    base_first_pass: dict[str, bool | None] = {case.id: None for case in suite}
    cand_first_pass: dict[str, bool | None] = {case.id: None for case in suite}

    # Index cases and scorers by case id for fast lookup.
    case_map: dict[str, TestCase] = {case.id: case for case in suite}
    scorers_map = {case.id: build_scorers(case, config.default_scorers) for case in suite}

    async def _collect(case_id: str, version: Version, start_idx: int, count: int) -> None:
        """Collect count samples starting at start_idx for one case and version."""
        nonlocal calls_used
        case = case_map[case_id]
        scorers = scorers_map[case_id]
        provider = get_provider(version)
        pass_map = base_pass_by_case if version is base_version else cand_pass_by_case
        n_map = base_n_by_case if version is base_version else cand_n_by_case
        first_map = base_first_pass if version is base_version else cand_first_pass

        for offset in range(count):
            sample_idx = start_idx + offset
            if calls_used >= budget_calls:
                break
            sample = await provider.generate(version, case, sample_idx)
            store.save_sample(run_id, sample)
            calls_used += 1
            passed = await _score_sample(sample, scorers, case)
            n_map[case_id] += 1
            if passed:
                pass_map[case_id] += 1
            if sample_idx == 0:
                first_map[case_id] = passed

    # Phase 1: collect k_min samples for every case.
    for case in suite:
        for version in [base_version, cand_version]:
            await _collect(case.id, version, start_idx=0, count=cfg.k_min)

    critical = {c.id for c in suite if c.critical}

    # Phase 2: adaptive rounds until settled, capped, or budget exhausted.
    while True:
        current_counts = _build_counts(
            suite,
            base_pass_by_case,
            base_n_by_case,
            cand_pass_by_case,
            cand_n_by_case,
            base_first_pass,
            cand_first_pass,
        )
        budget_left = budget_calls - calls_used
        schedule = next_round(current_counts, cfg, budget_left, critical)
        if not schedule:
            break
        for case_id, extra in schedule.items():
            base_start = base_n_by_case[case_id]
            cand_start = cand_n_by_case[case_id]
            await _collect(case_id, base_version, start_idx=base_start, count=extra)
            await _collect(case_id, cand_version, start_idx=cand_start, count=extra)

    item_counts = _build_counts(
        suite,
        base_pass_by_case,
        base_n_by_case,
        cand_pass_by_case,
        cand_n_by_case,
        base_first_pass,
        cand_first_pass,
    )

    # Items that never reached k_max are inconclusive when budget ran out.
    inconclusive_ids: set[str] = set()
    if calls_used >= budget_calls:
        for c in item_counts:
            if c.base_n < cfg.k_max:
                inconclusive_ids.add(c.case_id)

    item_results = classify_items(item_counts, alpha=config.gates.alpha, inconclusive_ids=inconclusive_ids)

    store.save_item_results(run_id, item_results)

    summary = compute_summary(item_counts, item_results, seed=42)
    summary = summary.model_copy(
        update={"slices": compute_slices(suite, item_counts, item_results)}
    )

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


def _build_counts(
    suite: list[TestCase],
    base_pass_by_case: dict[str, int],
    base_n_by_case: dict[str, int],
    cand_pass_by_case: dict[str, int],
    cand_n_by_case: dict[str, int],
    base_first_pass: dict[str, bool | None],
    cand_first_pass: dict[str, bool | None],
) -> list[ItemCounts]:
    """Build a list of ItemCounts from per-case accumulator dicts."""
    return [
        ItemCounts(
            case_id=case.id,
            base_pass=base_pass_by_case[case.id],
            base_n=base_n_by_case[case.id],
            cand_pass=cand_pass_by_case[case.id],
            cand_n=cand_n_by_case[case.id],
            base_first_pass=base_first_pass[case.id],
            cand_first_pass=cand_first_pass[case.id],
        )
        for case in suite
    ]


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
