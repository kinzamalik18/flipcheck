"""SQLite repository for FlipCheck.

This is the only module that writes SQL. All public methods are typed and
documented. Use sqlite3 from the stdlib only, no ORM.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from flipcheck.types import ItemClass, ItemCounts, ItemResult, Sample, TestCase

_SCHEMA_VERSION = 1

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS versions (
    version_id  TEXT PRIMARY KEY,
    name        TEXT,
    config_json TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    project      TEXT,
    suite_hash   TEXT,
    baseline_id  TEXT,
    candidate_id TEXT,
    config_json  TEXT,
    status       TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    calls_used   INTEGER,
    est_cost_usd REAL
);

CREATE TABLE IF NOT EXISTS cases (
    run_id       TEXT,
    case_id      TEXT,
    input_json   TEXT,
    expected_json TEXT,
    tags_json    TEXT,
    critical     INTEGER,
    PRIMARY KEY (run_id, case_id)
);

CREATE TABLE IF NOT EXISTS samples (
    sample_pk  TEXT PRIMARY KEY,
    run_id     TEXT,
    version_id TEXT,
    case_id    TEXT,
    sample_idx INTEGER,
    output     TEXT,
    error      TEXT,
    latency_ms INTEGER,
    tokens_in  INTEGER,
    tokens_out INTEGER,
    cached     INTEGER
);

CREATE TABLE IF NOT EXISTS scores (
    sample_pk TEXT,
    scorer    TEXT,
    value     REAL,
    passed    INTEGER,
    rationale TEXT
);

CREATE TABLE IF NOT EXISTS item_results (
    run_id        TEXT,
    case_id       TEXT,
    classification TEXT,
    base_pass     INTEGER,
    base_n        INTEGER,
    cand_pass     INTEGER,
    cand_n        INTEGER,
    p_regression  REAL,
    p_improvement REAL,
    PRIMARY KEY (run_id, case_id)
);

CREATE TABLE IF NOT EXISTS run_summaries (
    run_id       TEXT PRIMARY KEY,
    summary_json TEXT,
    verdict_json TEXT
);

CREATE TABLE IF NOT EXISTS output_cache (
    key         TEXT PRIMARY KEY,
    sample_json TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS judge_cache (
    key        TEXT PRIMARY KEY,
    score_json TEXT,
    created_at TEXT
);
"""


class SQLiteStore:
    """Thin SQLite repository. The only code in this package that writes SQL."""

    def __init__(self, path: str) -> None:
        """Open (or create) the SQLite database at path.

        Creates parent directories if they do not exist. On first open it
        creates all tables and sets PRAGMA user_version to 1.
        """
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._con = sqlite3.connect(str(db_path))
        self._con.row_factory = sqlite3.Row

        version: int = self._con.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            self._con.executescript(_CREATE_TABLES)
            self._con.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            self._con.commit()

    # ------------------------------------------------------------------
    # Samples
    # ------------------------------------------------------------------

    def save_sample(self, run_id: str, sample: Sample) -> None:
        """Persist one sample to the store.

        Also upserts a minimal row in the runs table so that list_runs can
        return this run even when save_run was never called.
        """
        sample_pk = (
            f"{run_id}:{sample.version_id}:{sample.case_id}:{sample.sample_idx}"
        )
        self._con.execute(
            """
            INSERT OR REPLACE INTO samples
                (sample_pk, run_id, version_id, case_id, sample_idx,
                 output, error, latency_ms, tokens_in, tokens_out, cached)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample_pk,
                run_id,
                sample.version_id,
                sample.case_id,
                sample.sample_idx,
                sample.output,
                sample.error,
                sample.latency_ms,
                sample.tokens_in,
                sample.tokens_out,
                int(sample.cached),
            ),
        )
        # Auto-register the run so list_runs works without an explicit save_run.
        # Use INSERT OR IGNORE so an existing row is not overwritten.
        # Use a Python-generated ISO timestamp with microsecond precision so
        # that ordering is correct even when two inserts happen within the same
        # second (SQLite's datetime('now') has only second resolution).
        now = datetime.now(tz=UTC).isoformat()
        self._con.execute(
            """
            INSERT OR IGNORE INTO runs
                (run_id, project, suite_hash, baseline_id, candidate_id,
                 config_json, status, started_at, finished_at,
                 calls_used, est_cost_usd)
            VALUES (?, NULL, NULL, NULL, NULL, NULL, 'running', ?, NULL, NULL, NULL)
            """,
            (run_id, now),
        )
        self._con.commit()

    def get_samples(
        self, run_id: str, case_id: str, version_id: str
    ) -> list[Sample]:
        """Return all samples for a given run, case and version.

        Returns an empty list when no matching rows exist.
        """
        rows = self._con.execute(
            """
            SELECT version_id, case_id, sample_idx, output, error,
                   latency_ms, tokens_in, tokens_out, cached
            FROM   samples
            WHERE  run_id = ? AND case_id = ? AND version_id = ?
            """,
            (run_id, case_id, version_id),
        ).fetchall()

        return [
            Sample(
                version_id=row["version_id"],
                case_id=row["case_id"],
                sample_idx=row["sample_idx"],
                output=row["output"],
                error=row["error"],
                latency_ms=row["latency_ms"],
                tokens_in=row["tokens_in"],
                tokens_out=row["tokens_out"],
                cached=bool(row["cached"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Item results
    # ------------------------------------------------------------------

    def save_item_results(self, run_id: str, results: list[ItemResult]) -> None:
        """Persist the list of ItemResult objects for a run.

        Uses INSERT OR REPLACE so re-running analysis overwrites stale rows.
        """
        self._con.executemany(
            """
            INSERT OR REPLACE INTO item_results
                (run_id, case_id, classification,
                 base_pass, base_n, cand_pass, cand_n,
                 p_regression, p_improvement)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    r.counts.case_id,
                    str(r.classification),
                    r.counts.base_pass,
                    r.counts.base_n,
                    r.counts.cand_pass,
                    r.counts.cand_n,
                    r.p_regression,
                    r.p_improvement,
                )
                for r in results
            ],
        )
        self._con.commit()

    def get_item_results(self, run_id: str) -> list[ItemResult]:
        """Return all ItemResult objects stored for a run.

        Returns an empty list when no matching rows exist.
        """
        rows = self._con.execute(
            """
            SELECT case_id, classification,
                   base_pass, base_n, cand_pass, cand_n,
                   p_regression, p_improvement
            FROM   item_results
            WHERE  run_id = ?
            """,
            (run_id,),
        ).fetchall()

        return [
            ItemResult(
                counts=ItemCounts(
                    case_id=row["case_id"],
                    base_pass=row["base_pass"],
                    base_n=row["base_n"],
                    cand_pass=row["cand_pass"],
                    cand_n=row["cand_n"],
                ),
                classification=ItemClass(row["classification"]),
                p_regression=row["p_regression"],
                p_improvement=row["p_improvement"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def save_run(
        self,
        run_id: str,
        project: str,
        suite_hash: str,
        baseline_id: str,
        candidate_id: str,
        config_json: str,
        status: str,
        started_at: str,
    ) -> None:
        """Insert a new run row.

        Uses INSERT OR REPLACE so calling save_run twice for the same id is safe.
        """
        self._con.execute(
            """
            INSERT OR REPLACE INTO runs
                (run_id, project, suite_hash, baseline_id, candidate_id,
                 config_json, status, started_at, finished_at,
                 calls_used, est_cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
            """,
            (
                run_id,
                project,
                suite_hash,
                baseline_id,
                candidate_id,
                config_json,
                status,
                started_at,
            ),
        )
        self._con.commit()

    def update_run_finished(
        self,
        run_id: str,
        status: str,
        finished_at: str,
        calls_used: int,
        est_cost_usd: float | None,
    ) -> None:
        """Update the finished_at, status, calls_used and est_cost_usd columns."""
        self._con.execute(
            """
            UPDATE runs
            SET status = ?, finished_at = ?, calls_used = ?, est_cost_usd = ?
            WHERE run_id = ?
            """,
            (status, finished_at, calls_used, est_cost_usd, run_id),
        )
        self._con.commit()

    def list_runs(self) -> list[dict]:
        """Return a list of run metadata dicts, most recent first.

        Rows are ordered by started_at descending so the most recent run
        appears first.
        """
        rows = self._con.execute(
            "SELECT * FROM runs ORDER BY started_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Run summaries
    # ------------------------------------------------------------------

    def save_run_summary(
        self, run_id: str, summary_json: str, verdict_json: str
    ) -> None:
        """Persist a serialized RunSummary and Verdict for a run."""
        self._con.execute(
            """
            INSERT OR REPLACE INTO run_summaries (run_id, summary_json, verdict_json)
            VALUES (?, ?, ?)
            """,
            (run_id, summary_json, verdict_json),
        )
        self._con.commit()

    def get_run_summary(self, run_id: str) -> dict | None:
        """Return the stored summary and verdict for a run, or None if not found."""
        row = self._con.execute(
            "SELECT summary_json, verdict_json FROM run_summaries WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    # ------------------------------------------------------------------
    # Versions
    # ------------------------------------------------------------------

    def save_version(
        self, version_id: str, name: str, config_json: str, created_at: str
    ) -> None:
        """Save a version record.

        Uses INSERT OR REPLACE so calling this twice for the same id is safe.
        """
        self._con.execute(
            """
            INSERT OR REPLACE INTO versions (version_id, name, config_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (version_id, name, config_json, created_at),
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Cases
    # ------------------------------------------------------------------

    def save_case(self, run_id: str, case: TestCase) -> None:
        """Save a test case record for a run.

        Uses INSERT OR REPLACE so re-saving the same case is safe.
        """
        self._con.execute(
            """
            INSERT OR REPLACE INTO cases
                (run_id, case_id, input_json, expected_json, tags_json, critical)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                case.id,
                json.dumps(case.input),
                json.dumps(case.expected),
                json.dumps(case.tags),
                int(case.critical),
            ),
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def save_score(
        self,
        sample_pk: str,
        scorer: str,
        value: float,
        passed: bool,
        rationale: str | None,
    ) -> None:
        """Save a score record.

        Inserts a new row each time; scores are not deduplicated by key.
        """
        self._con.execute(
            """
            INSERT INTO scores (sample_pk, scorer, value, passed, rationale)
            VALUES (?, ?, ?, ?, ?)
            """,
            (sample_pk, scorer, value, int(passed), rationale),
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying sqlite3 connection."""
        self._con.close()
