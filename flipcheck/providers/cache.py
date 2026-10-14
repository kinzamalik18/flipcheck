"""Output cache backed by SQLite.

Covers SPEC section 6 (output cache). The cache key is
sha256(version_id + case_id + str(sample_idx)).
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime

from flipcheck.types import Sample


class OutputCache:
    """Persistent output cache stored in a SQLite database.

    Each entry is keyed by sha256(version_id + case_id + str(sample_idx))
    and stores the full Sample as JSON. Samples retrieved from cache have
    their cached field set to True.
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS output_cache (
        key        TEXT PRIMARY KEY,
        sample_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """

    def __init__(self, path: str) -> None:
        """Open or create the SQLite database at path and ensure the table exists."""
        self._path = path
        con = self._connect()
        con.execute(self._CREATE_TABLE)
        con.commit()
        con.close()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path)
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def _key(version_id: str, case_id: str, sample_idx: int) -> str:
        """Return the sha256 hex digest used as the cache key."""
        raw = (version_id + case_id + str(sample_idx)).encode()
        return hashlib.sha256(raw).hexdigest()

    def get(self, version_id: str, case_id: str, sample_idx: int) -> Sample | None:
        """Return the cached Sample, or None on a cache miss.

        A returned Sample always has cached=True.
        """
        key = self._key(version_id, case_id, sample_idx)
        con = self._connect()
        row = con.execute(
            "SELECT sample_json FROM output_cache WHERE key = ?", (key,)
        ).fetchone()
        con.close()
        if row is None:
            return None
        sample = Sample.model_validate_json(row["sample_json"])
        return sample.model_copy(update={"cached": True})

    def store(self, version_id: str, case_id: str, sample_idx: int, sample: Sample) -> None:
        """Persist a sample in the cache.

        If an entry for this key already exists it is replaced.
        """
        key = self._key(version_id, case_id, sample_idx)
        now = datetime.now(tz=UTC).isoformat()
        con = self._connect()
        con.execute(
            "INSERT OR REPLACE INTO output_cache (key, sample_json, created_at) VALUES (?, ?, ?)",
            (key, sample.model_dump_json(), now),
        )
        con.commit()
        con.close()
