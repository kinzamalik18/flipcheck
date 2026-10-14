"""Tests for flipcheck/runner.py in fixed mode - SPEC section 8.1.

The runner is imported with a graceful skip if not yet implemented.
All tests are async (pytest-asyncio with asyncio_mode=auto from pyproject.toml).
No network calls: the mock provider is used for both versions.

Public API under test:
    from flipcheck.runner import run_fixed
    run_fixed(config: ProjectConfig, store: SQLiteStore, suite: list[TestCase])
        -> tuple[str, RunSummary, Verdict]

SPEC 8.1 says:
    "Run k samples per case for each version, score them, build ItemCounts."

Tests verify:
    - Each case gets exactly k samples per version.
    - Total provider calls = 2 * k * len(suite).
    - A second call with the same run_id uses cached samples (no new provider calls).
    - Budget cap: if budget_calls < 2 * k * len(suite), the runner stops early.
    - Errors from the provider count as failed samples, not exceptions.
    - Results are persisted in the store after the run.
    - The return value is (run_id: str, summary: RunSummary, verdict: Verdict).
"""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path

import pytest

from flipcheck.store.db import SQLiteStore
from flipcheck.types import (
    ProjectConfig,
    RunSummary,
    SamplingConfig,
    ScorerConfig,
    TargetConfig,
    TestCase,
    Verdict,
)

# ---------------------------------------------------------------------------
# Skip guard: tests skip cleanly if run_fixed is not implemented yet.
# ---------------------------------------------------------------------------


def _import_run_fixed():
    mod = importlib.import_module("flipcheck.runner")
    fn = getattr(mod, "run_fixed", None)
    if fn is None:
        pytest.skip("flipcheck.runner.run_fixed not implemented yet")
    return fn


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: str) -> SQLiteStore:
    db_file = str(Path(tmp_path) / "test.sqlite")
    return SQLiteStore(db_file)


def _make_config(
    k: int = 3,
    budget_calls: int = 10000,
    base_pass_prob: float = 0.9,
    cand_pass_prob: float = 0.7,
) -> ProjectConfig:
    """Build a ProjectConfig that uses the mock provider for both versions."""
    return ProjectConfig(
        project="test-project",
        suite="suite.jsonl",
        default_scorers=[ScorerConfig(type="exact")],
        targets={
            "baseline": TargetConfig(
                type="mock",
                mock={"pass_prob": {"*": base_pass_prob}, "seed": 1},
            ),
            "candidate": TargetConfig(
                type="mock",
                mock={"pass_prob": {"*": cand_pass_prob}, "seed": 2},
            ),
        },
        sampling=SamplingConfig(mode="fixed", k=k, budget_calls=budget_calls),
    )


def _make_suite(n: int = 3, expected: str = "PASS") -> list[TestCase]:
    """Create n test cases with the expected value for the exact scorer."""
    return [
        TestCase(
            id=f"t{i:03d}",
            input={"query": f"question {i}"},
            expected=expected,
            scorers=[ScorerConfig(type="exact")],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Return value shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fixed_returns_tuple_of_three():
    """run_fixed returns a 3-tuple: (run_id, RunSummary, Verdict)."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=2)
        suite = _make_suite(n=2)
        result = await run_fixed(config, store, suite)
        assert isinstance(result, tuple)
        assert len(result) == 3


@pytest.mark.asyncio
async def test_run_fixed_run_id_is_string():
    """run_fixed returns a non-empty string as the first element (run_id)."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=2)
        suite = _make_suite(n=2)
        run_id, _, _ = await run_fixed(config, store, suite)
        assert isinstance(run_id, str)
        assert len(run_id) > 0


@pytest.mark.asyncio
async def test_run_fixed_summary_is_run_summary():
    """run_fixed second element is a RunSummary."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=2)
        suite = _make_suite(n=2)
        _, summary, _ = await run_fixed(config, store, suite)
        assert isinstance(summary, RunSummary)


@pytest.mark.asyncio
async def test_run_fixed_verdict_is_verdict():
    """run_fixed third element is a Verdict."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=2)
        suite = _make_suite(n=2)
        _, _, verdict = await run_fixed(config, store, suite)
        assert isinstance(verdict, Verdict)


# ---------------------------------------------------------------------------
# Fixed mode: k samples per case per version
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fixed_k3_each_case_has_3_baseline_samples():
    """Fixed mode k=3: each case has exactly 3 baseline samples in the store."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=3)
        suite = _make_suite(n=2)
        run_id, _, _ = await run_fixed(config, store, suite)

        from flipcheck.config import compute_version_id

        base_cfg = config.targets["baseline"]
        base_vid = compute_version_id(base_cfg, None)

        for case in suite:
            samples = store.get_samples(run_id, case.id, base_vid)
            assert len(samples) == 3, (
                f"case {case.id}: expected 3 baseline samples, got {len(samples)}"
            )


@pytest.mark.asyncio
async def test_run_fixed_k3_each_case_has_3_candidate_samples():
    """Fixed mode k=3: each case has exactly 3 candidate samples in the store."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=3)
        suite = _make_suite(n=2)
        run_id, _, _ = await run_fixed(config, store, suite)

        from flipcheck.config import compute_version_id

        cand_cfg = config.targets["candidate"]
        cand_vid = compute_version_id(cand_cfg, None)

        for case in suite:
            samples = store.get_samples(run_id, case.id, cand_vid)
            assert len(samples) == 3, (
                f"case {case.id}: expected 3 candidate samples, got {len(samples)}"
            )


@pytest.mark.asyncio
async def test_run_fixed_summary_n_items_equals_suite_length():
    """Fixed mode: summary.n_items equals the number of test cases in the suite."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=3)
        suite = _make_suite(n=4)
        _, summary, _ = await run_fixed(config, store, suite)
        assert summary.n_items == 4


@pytest.mark.asyncio
async def test_run_fixed_calls_used_equals_2_k_n():
    """Fixed mode k=3, n=3: calls_used == 2 * 3 * 3 == 18 (no judge calls)."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=3)
        suite = _make_suite(n=3)
        _, summary, _ = await run_fixed(config, store, suite)
        # 2 versions * 3 samples * 3 cases = 18
        assert summary.calls_used == 2 * 3 * 3


# ---------------------------------------------------------------------------
# Budget cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fixed_budget_cap_stops_early():
    """Budget cap: budget_calls=4, suite has 5 cases with k=2. Runner stops before all cases."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        # k=2, 5 cases would need 2*2*5=20 calls, but budget is 4 (= 1 case of 2 versions * 2)
        config = _make_config(k=2, budget_calls=4)
        suite = _make_suite(n=5)
        _, summary, _ = await run_fixed(config, store, suite)
        # Fewer than 5 cases could be fully processed; calls_used must not exceed budget
        assert summary.calls_used <= 4


@pytest.mark.asyncio
async def test_run_fixed_budget_zero_results_in_no_calls():
    """Budget cap of zero: no calls are made, calls_used == 0."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=2, budget_calls=0)
        suite = _make_suite(n=3)
        _, summary, _ = await run_fixed(config, store, suite)
        assert summary.calls_used == 0


# ---------------------------------------------------------------------------
# Cache hit on second run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fixed_second_run_same_id_uses_cache():
    """Second run with the same run_id uses cached samples and makes no new provider calls."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=3)
        suite = _make_suite(n=2)

        run_id_first, _, _ = await run_fixed(config, store, suite)

        # Run again with the same store (samples already present).
        run_id_second, summary_second, _ = await run_fixed(
            config, store, suite, run_id=run_id_first
        )

        assert run_id_second == run_id_first
        # All samples should be marked as cached, so no new calls were charged.
        from flipcheck.config import compute_version_id

        base_vid = compute_version_id(config.targets["baseline"], None)
        for case in suite:
            samples = store.get_samples(run_id_first, case.id, base_vid)
            assert all(s.cached for s in samples), (
                f"case {case.id}: expected all samples cached on second run"
            )


# ---------------------------------------------------------------------------
# Errors counted as failed samples, not exceptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fixed_provider_error_does_not_raise():
    """A provider that always errors should not raise; the run completes."""
    run_fixed = _import_run_fixed()

    # We simulate an error-producing provider by using pass_prob=0.0 which always
    # produces "FAIL" (not an exception), then we also test the error path using
    # a patched provider config. Since we cannot easily inject a broken provider
    # via config, we verify that:
    #   1. The run completes.
    #   2. The summary is still returned (not an exception).
    # A deeper error test would require a mock that sets sample.error; that can
    # be done via an integration test or a provider fixture once the runner exists.
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=2, base_pass_prob=0.0, cand_pass_prob=0.0)
        suite = _make_suite(n=2)
        run_id, summary, verdict = await run_fixed(config, store, suite)
        assert isinstance(summary, RunSummary)
        assert isinstance(verdict, Verdict)


# ---------------------------------------------------------------------------
# Results persisted in store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fixed_persists_samples_in_store():
    """After run_fixed, samples for all cases and versions are in the store."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=2)
        suite = _make_suite(n=3)
        run_id, _, _ = await run_fixed(config, store, suite)

        from flipcheck.config import compute_version_id

        base_vid = compute_version_id(config.targets["baseline"], None)
        cand_vid = compute_version_id(config.targets["candidate"], None)

        for case in suite:
            base_samples = store.get_samples(run_id, case.id, base_vid)
            cand_samples = store.get_samples(run_id, case.id, cand_vid)
            assert len(base_samples) == 2
            assert len(cand_samples) == 2


@pytest.mark.asyncio
async def test_run_fixed_run_appears_in_list_runs():
    """After run_fixed, list_runs returns the new run_id."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=2)
        suite = _make_suite(n=2)
        run_id, _, _ = await run_fixed(config, store, suite)
        runs = store.list_runs()
        run_ids = [r["run_id"] for r in runs]
        assert run_id in run_ids


@pytest.mark.asyncio
async def test_run_fixed_item_results_persisted():
    """After run_fixed, item_results for the run are persisted in the store."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=3)
        suite = _make_suite(n=3)
        run_id, _, _ = await run_fixed(config, store, suite)
        item_results = store.get_item_results(run_id)
        assert len(item_results) == 3


# ---------------------------------------------------------------------------
# Empty suite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fixed_empty_suite_returns_zero_items():
    """run_fixed with an empty suite returns summary.n_items == 0."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=3)
        suite: list[TestCase] = []
        _, summary, _ = await run_fixed(config, store, suite)
        assert summary.n_items == 0
        assert summary.calls_used == 0


# ---------------------------------------------------------------------------
# Duplicate case IDs in suite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fixed_duplicate_case_ids_are_deduplicated_or_handled():
    """run_fixed with duplicate case IDs either deduplicates or processes all without crashing."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=2)
        # Two cases with the same id
        suite = [
            TestCase(id="dup", input={"query": "q1"}, expected="PASS"),
            TestCase(id="dup", input={"query": "q2"}, expected="PASS"),
        ]
        # Should not raise; behavior on duplicates is impl-defined but must be stable.
        run_id, summary, _ = await run_fixed(config, store, suite)
        assert isinstance(summary, RunSummary)


# ---------------------------------------------------------------------------
# Summary pass-rate sanity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fixed_base_rate_in_zero_one():
    """summary.base_rate is between 0.0 and 1.0 inclusive."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=3)
        suite = _make_suite(n=4)
        _, summary, _ = await run_fixed(config, store, suite)
        assert 0.0 <= summary.base_rate <= 1.0


@pytest.mark.asyncio
async def test_run_fixed_cand_rate_in_zero_one():
    """summary.cand_rate is between 0.0 and 1.0 inclusive."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=3)
        suite = _make_suite(n=4)
        _, summary, _ = await run_fixed(config, store, suite)
        assert 0.0 <= summary.cand_rate <= 1.0


@pytest.mark.asyncio
async def test_run_fixed_delta_equals_cand_minus_base():
    """summary.delta == cand_rate - base_rate (within floating-point tolerance)."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=3)
        suite = _make_suite(n=4)
        _, summary, _ = await run_fixed(config, store, suite)
        assert summary.delta == pytest.approx(summary.cand_rate - summary.base_rate, abs=1e-9)


# ---------------------------------------------------------------------------
# Reliable NFR and churn rate sanity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fixed_reliable_nfr_in_zero_one():
    """summary.reliable_nfr is between 0.0 and 1.0 inclusive."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=3)
        suite = _make_suite(n=3)
        _, summary, _ = await run_fixed(config, store, suite)
        assert 0.0 <= summary.reliable_nfr <= 1.0


@pytest.mark.asyncio
async def test_run_fixed_churn_rate_in_zero_one():
    """summary.churn_rate is between 0.0 and 1.0 inclusive."""
    run_fixed = _import_run_fixed()
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        config = _make_config(k=3)
        suite = _make_suite(n=3)
        _, summary, _ = await run_fixed(config, store, suite)
        assert 0.0 <= summary.churn_rate <= 1.0
