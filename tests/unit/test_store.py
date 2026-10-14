"""Tests for flipcheck/store/db.py: SQLite store.

Covers SPEC section 11.
"""

from __future__ import annotations

import time

import pytest

import flipcheck.store.db as _store_mod
from flipcheck.types import (
    ItemClass,
    ItemCounts,
    ItemResult,
    Sample,
)

if not hasattr(_store_mod, "SQLiteStore"):
    pytest.skip("flipcheck.store.db.SQLiteStore not implemented yet", allow_module_level=True)

SQLiteStore = _store_mod.SQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    case_id: str = "t001",
    version_id: str = "abc123def456",
    sample_idx: int = 0,
    output: str | None = "PASS",
) -> Sample:
    return Sample(
        case_id=case_id,
        version_id=version_id,
        sample_idx=sample_idx,
        output=output,
    )


def _make_item_result(case_id: str = "t001") -> ItemResult:
    counts = ItemCounts(
        case_id=case_id,
        base_pass=4,
        base_n=5,
        cand_pass=2,
        cand_n=5,
    )
    return ItemResult(
        counts=counts,
        classification=ItemClass.RELIABLE_REGRESSION,
        p_regression=0.02,
        p_improvement=None,
    )


# ---------------------------------------------------------------------------
# Store creation and schema
# ---------------------------------------------------------------------------


def test_store_opens_without_error(tmp_path):
    path = str(tmp_path / "test.sqlite")
    store = SQLiteStore(path)
    assert store is not None


def test_store_creates_file(tmp_path):
    path = tmp_path / "test.sqlite"
    SQLiteStore(str(path))
    assert path.exists()


def test_store_tables_exist(tmp_path):
    """Verify that the expected tables are created on first open."""
    import sqlite3

    path = str(tmp_path / "test.sqlite")
    SQLiteStore(path)
    con = sqlite3.connect(path)
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    con.close()
    for expected in ("samples", "runs", "item_results"):
        assert expected in tables, f"table {expected!r} not found"


# ---------------------------------------------------------------------------
# save_sample / get_samples
# ---------------------------------------------------------------------------


def test_save_sample_and_get_it_back(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.sqlite"))
    sample = _make_sample()
    store.save_sample("run-1", sample)
    results = store.get_samples("run-1", "t001", "abc123def456")
    assert len(results) == 1


def test_get_samples_returns_sample_object(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.sqlite"))
    sample = _make_sample()
    store.save_sample("run-1", sample)
    results = store.get_samples("run-1", "t001", "abc123def456")
    assert isinstance(results[0], Sample)


def test_get_samples_correct_fields(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.sqlite"))
    sample = _make_sample(output="PASS")
    store.save_sample("run-1", sample)
    results = store.get_samples("run-1", "t001", "abc123def456")
    assert results[0].output == "PASS"
    assert results[0].case_id == "t001"
    assert results[0].version_id == "abc123def456"


def test_get_samples_wrong_run_id_returns_empty(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.sqlite"))
    store.save_sample("run-1", _make_sample())
    results = store.get_samples("run-999", "t001", "abc123def456")
    assert results == []


def test_get_samples_wrong_case_id_returns_empty(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.sqlite"))
    store.save_sample("run-1", _make_sample(case_id="t001"))
    results = store.get_samples("run-1", "t999", "abc123def456")
    assert results == []


def test_get_samples_wrong_version_id_returns_empty(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.sqlite"))
    store.save_sample("run-1", _make_sample(version_id="abc123def456"))
    results = store.get_samples("run-1", "t001", "000000000000")
    assert results == []


def test_multiple_samples_returned(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.sqlite"))
    for idx in range(3):
        store.save_sample("run-1", _make_sample(sample_idx=idx))
    results = store.get_samples("run-1", "t001", "abc123def456")
    assert len(results) == 3


# ---------------------------------------------------------------------------
# save_item_results
# ---------------------------------------------------------------------------


def test_save_item_results_no_error(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.sqlite"))
    results = [_make_item_result("t001"), _make_item_result("t002")]
    store.save_item_results("run-1", results)


def test_save_item_results_can_be_retrieved(tmp_path):
    """After saving, the data must be queryable (at minimum via list_runs or a
    dedicated retrieval method if exposed). The store protocol guarantees
    save_item_results; retrieval details are implementation-specific but the
    test verifies the call does not silently drop data by checking the SQLite
    table directly."""
    import sqlite3

    path = str(tmp_path / "test.sqlite")
    store = SQLiteStore(path)
    results = [_make_item_result("t001"), _make_item_result("t002")]
    store.save_item_results("run-1", results)

    con = sqlite3.connect(path)
    rows = con.execute("SELECT case_id FROM item_results WHERE run_id = 'run-1'").fetchall()
    con.close()
    case_ids = {r[0] for r in rows}
    assert "t001" in case_ids
    assert "t002" in case_ids


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


def test_list_runs_returns_list(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.sqlite"))
    result = store.list_runs()
    assert isinstance(result, list)


def test_list_runs_empty_when_no_runs(tmp_path):
    store = SQLiteStore(str(tmp_path / "test.sqlite"))
    assert store.list_runs() == []


def test_list_runs_reverse_chronological_order(tmp_path):
    """Runs must come back most recent first."""
    store = SQLiteStore(str(tmp_path / "test.sqlite"))

    # Save two samples under different run_ids so the store knows about them.
    # Small sleep ensures distinct timestamps on systems where resolution is low.
    store.save_sample("run-early", _make_sample(case_id="t001"))
    time.sleep(0.05)
    store.save_sample("run-late", _make_sample(case_id="t002"))

    runs = store.list_runs()
    run_ids = [r["run_id"] if isinstance(r, dict) else r for r in runs]
    # "run-late" must appear before "run-early"
    assert run_ids.index("run-late") < run_ids.index("run-early")


# ---------------------------------------------------------------------------
# Persistence: second open does not reset data
# ---------------------------------------------------------------------------


def test_data_persists_across_store_instances(tmp_path):
    path = str(tmp_path / "test.sqlite")
    store1 = SQLiteStore(path)
    store1.save_sample("run-1", _make_sample())

    store2 = SQLiteStore(path)
    results = store2.get_samples("run-1", "t001", "abc123def456")
    assert len(results) == 1


def test_second_open_does_not_wipe_item_results(tmp_path):
    import sqlite3

    path = str(tmp_path / "test.sqlite")
    store1 = SQLiteStore(path)
    store1.save_item_results("run-1", [_make_item_result("t001")])

    SQLiteStore(path)  # second open
    con = sqlite3.connect(path)
    count = con.execute("SELECT COUNT(*) FROM item_results WHERE run_id = 'run-1'").fetchone()[0]
    con.close()
    assert count == 1
