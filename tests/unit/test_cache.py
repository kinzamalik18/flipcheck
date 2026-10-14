"""Tests for flipcheck/providers/cache.py: output cache.

Covers SPEC section 6 (output cache keyed by sha256(version_id, case_id, sample_idx)).
"""

from __future__ import annotations

import hashlib

import pytest

import flipcheck.providers.cache as _cache_mod
from flipcheck.types import Sample

if not hasattr(_cache_mod, "OutputCache"):
    pytest.skip("flipcheck.providers.cache.OutputCache not implemented yet", allow_module_level=True)

OutputCache = _cache_mod.OutputCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(
    case_id: str = "t001",
    version_id: str = "abc123def456",
    sample_idx: int = 0,
    output: str = "PASS",
) -> Sample:
    return Sample(
        case_id=case_id,
        version_id=version_id,
        sample_idx=sample_idx,
        output=output,
    )


def _expected_key(version_id: str, case_id: str, sample_idx: int) -> str:
    raw = version_id + case_id + str(sample_idx)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Cache miss
# ---------------------------------------------------------------------------


def test_cache_miss_returns_none(tmp_path):
    cache = OutputCache(str(tmp_path / "cache.sqlite"))
    result = cache.get("abc123def456", "t001", 0)
    assert result is None


def test_cache_miss_for_unknown_key(tmp_path):
    cache = OutputCache(str(tmp_path / "cache.sqlite"))
    result = cache.get("nonexistent_ver", "t999", 7)
    assert result is None


# ---------------------------------------------------------------------------
# Cache hit
# ---------------------------------------------------------------------------


def test_cache_hit_returns_sample(tmp_path):
    cache = OutputCache(str(tmp_path / "cache.sqlite"))
    sample = _make_sample()
    cache.store("abc123def456", "t001", 0, sample)
    result = cache.get("abc123def456", "t001", 0)
    assert result is not None


def test_cache_hit_correct_output(tmp_path):
    cache = OutputCache(str(tmp_path / "cache.sqlite"))
    sample = _make_sample(output="FAIL")
    cache.store("abc123def456", "t001", 0, sample)
    result = cache.get("abc123def456", "t001", 0)
    assert result.output == "FAIL"


def test_cache_hit_correct_case_id(tmp_path):
    cache = OutputCache(str(tmp_path / "cache.sqlite"))
    sample = _make_sample(case_id="t007")
    cache.store("abc123def456", "t007", 0, sample)
    result = cache.get("abc123def456", "t007", 0)
    assert result.case_id == "t007"


def test_cache_hit_sets_cached_true(tmp_path):
    """A sample returned from cache must have cached=True."""
    cache = OutputCache(str(tmp_path / "cache.sqlite"))
    sample = _make_sample(output="PASS")
    cache.store("abc123def456", "t001", 0, sample)
    result = cache.get("abc123def456", "t001", 0)
    assert result.cached is True


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


def test_cache_key_is_sha256_of_version_case_idx(tmp_path):
    """The cache must use sha256(version_id + case_id + str(sample_idx)) as key."""
    import sqlite3

    path = str(tmp_path / "cache.sqlite")
    cache = OutputCache(path)
    sample = _make_sample(version_id="ver0", case_id="t001", sample_idx=2)
    cache.store("ver0", "t001", 2, sample)

    expected_key = _expected_key("ver0", "t001", 2)
    con = sqlite3.connect(path)
    rows = con.execute("SELECT key FROM output_cache").fetchall()
    con.close()
    keys = {r[0] for r in rows}
    assert expected_key in keys, f"expected key {expected_key!r} not found in {keys}"


def test_different_version_id_is_cache_miss(tmp_path):
    cache = OutputCache(str(tmp_path / "cache.sqlite"))
    sample = _make_sample(version_id="ver_A")
    cache.store("ver_A", "t001", 0, sample)
    result = cache.get("ver_B", "t001", 0)
    assert result is None


def test_different_case_id_is_cache_miss(tmp_path):
    cache = OutputCache(str(tmp_path / "cache.sqlite"))
    sample = _make_sample(case_id="t001")
    cache.store("abc123def456", "t001", 0, sample)
    result = cache.get("abc123def456", "t002", 0)
    assert result is None


def test_different_sample_idx_is_cache_miss(tmp_path):
    cache = OutputCache(str(tmp_path / "cache.sqlite"))
    sample = _make_sample(sample_idx=0)
    cache.store("abc123def456", "t001", 0, sample)
    result = cache.get("abc123def456", "t001", 1)
    assert result is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_cache_persists_across_instances(tmp_path):
    path = str(tmp_path / "cache.sqlite")
    c1 = OutputCache(path)
    c1.store("abc123def456", "t001", 0, _make_sample())
    c2 = OutputCache(path)
    assert c2.get("abc123def456", "t001", 0) is not None
