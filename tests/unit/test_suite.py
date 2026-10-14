"""Tests for flipcheck/suite.py: suite loading and import from logs.

Covers SPEC section 5.
"""

from __future__ import annotations

import csv
import json

import pytest
import yaml

import flipcheck.suite as _suite_mod
from flipcheck.types import TestCase

if not hasattr(_suite_mod, "load_suite"):
    pytest.skip("flipcheck.suite.load_suite not implemented yet", allow_module_level=True)

load_suite = _suite_mod.load_suite
import_from_logs = _suite_mod.import_from_logs

FIXTURES_DIR = "C:/Users/HP/Documents/flipcheck/tests/fixtures"


# ---------------------------------------------------------------------------
# load_suite: JSONL
# ---------------------------------------------------------------------------


def test_load_jsonl_returns_list_of_test_cases():
    cases = load_suite(f"{FIXTURES_DIR}/small.jsonl")
    assert all(isinstance(c, TestCase) for c in cases)


def test_load_jsonl_returns_five_cases():
    cases = load_suite(f"{FIXTURES_DIR}/small.jsonl")
    assert len(cases) == 5


def test_load_jsonl_first_case_id():
    cases = load_suite(f"{FIXTURES_DIR}/small.jsonl")
    assert cases[0].id == "t001"


def test_load_jsonl_critical_flag():
    cases = load_suite(f"{FIXTURES_DIR}/small.jsonl")
    # t003 is marked critical in the fixture
    by_id = {c.id: c for c in cases}
    assert by_id["t003"].critical is True


def test_load_jsonl_tags_populated():
    cases = load_suite(f"{FIXTURES_DIR}/small.jsonl")
    by_id = {c.id: c for c in cases}
    assert "urdu" in by_id["t003"].tags
    assert "emergency" in by_id["t003"].tags


def test_load_jsonl_expected_value():
    cases = load_suite(f"{FIXTURES_DIR}/small.jsonl")
    assert cases[0].expected == "PASS"


def test_load_jsonl_input_is_dict():
    cases = load_suite(f"{FIXTURES_DIR}/small.jsonl")
    assert isinstance(cases[0].input, dict)


# ---------------------------------------------------------------------------
# load_suite: CSV
# ---------------------------------------------------------------------------


def _write_csv(tmp_path, rows: list[dict], filename: str = "suite.csv") -> str:
    path = tmp_path / filename
    if rows:
        fieldnames = list(rows[0].keys())
        with open(str(path), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return str(path)


def test_load_csv_returns_test_cases(tmp_path):
    rows = [
        {
            "id": "c001",
            "input_query": "hello",
            "expected": "PASS",
            "tags": "greeting",
            "critical": "false",
        },
        {
            "id": "c002",
            "input_query": "bye",
            "expected": "PASS",
            "tags": "",
            "critical": "false",
        },
    ]
    path = _write_csv(tmp_path, rows)
    cases = load_suite(path)
    assert all(isinstance(c, TestCase) for c in cases)


def test_load_csv_case_count(tmp_path):
    rows = [
        {"id": f"c{i:03d}", "input_query": f"q{i}", "expected": "PASS", "tags": "", "critical": "false"}
        for i in range(3)
    ]
    path = _write_csv(tmp_path, rows)
    cases = load_suite(path)
    assert len(cases) == 3


def test_load_csv_id_field(tmp_path):
    rows = [{"id": "csv001", "input_query": "hi", "expected": "yes", "tags": "", "critical": "false"}]
    path = _write_csv(tmp_path, rows)
    cases = load_suite(path)
    assert cases[0].id == "csv001"


def test_load_csv_critical_true(tmp_path):
    rows = [{"id": "c001", "input_query": "hi", "expected": "yes", "tags": "a", "critical": "true"}]
    path = _write_csv(tmp_path, rows)
    cases = load_suite(path)
    assert cases[0].critical is True


# ---------------------------------------------------------------------------
# load_suite: YAML
# ---------------------------------------------------------------------------


def _write_yaml_suite(tmp_path, cases_data: list[dict], filename: str = "suite.yaml") -> str:
    path = tmp_path / filename
    path.write_text(yaml.dump(cases_data), encoding="utf-8")
    return str(path)


def test_load_yaml_returns_test_cases(tmp_path):
    data = [
        {"id": "y001", "input": {"query": "hello"}, "expected": "PASS"},
        {"id": "y002", "input": {"query": "world"}, "expected": "PASS"},
    ]
    path = _write_yaml_suite(tmp_path, data)
    cases = load_suite(path)
    assert all(isinstance(c, TestCase) for c in cases)


def test_load_yaml_case_count(tmp_path):
    data = [{"id": f"y{i:03d}", "input": {"query": f"q{i}"}} for i in range(4)]
    path = _write_yaml_suite(tmp_path, data)
    cases = load_suite(path)
    assert len(cases) == 4


def test_load_yaml_id_field(tmp_path):
    data = [{"id": "yaml001", "input": {"query": "hi"}}]
    path = _write_yaml_suite(tmp_path, data)
    cases = load_suite(path)
    assert cases[0].id == "yaml001"


# ---------------------------------------------------------------------------
# load_suite: error cases
# ---------------------------------------------------------------------------


def _write_jsonl(tmp_path, lines: list[dict], filename: str = "suite.jsonl") -> str:
    path = tmp_path / filename
    path.write_text("\n".join(json.dumps(row) for row in lines), encoding="utf-8")
    return str(path)


def test_duplicate_ids_raise_error(tmp_path):
    lines = [
        {"id": "dup", "input": {"query": "a"}, "expected": "PASS"},
        {"id": "dup", "input": {"query": "b"}, "expected": "PASS"},
    ]
    path = _write_jsonl(tmp_path, lines)
    with pytest.raises(Exception, match=r"(?i)(duplicate|id)"):
        load_suite(path)


def test_unknown_scorer_type_raises_error_at_load_time(tmp_path):
    lines = [
        {
            "id": "t001",
            "input": {"query": "hello"},
            "expected": "PASS",
            "scorers": [{"type": "magic"}],
        }
    ]
    path = _write_jsonl(tmp_path, lines)
    with pytest.raises((ValueError, KeyError, TypeError, SystemExit)):
        load_suite(path)


def test_missing_id_field_raises_clear_error(tmp_path):
    lines = [{"input": {"query": "hello"}, "expected": "PASS"}]
    path = _write_jsonl(tmp_path, lines)
    with pytest.raises((ValueError, KeyError, TypeError, SystemExit)):
        load_suite(path)


# ---------------------------------------------------------------------------
# import_from_logs
# ---------------------------------------------------------------------------


def test_import_from_logs_returns_test_cases():
    records = [
        {"input_text": "hello world", "label": "PASS"},
        {"input_text": "bye", "label": "FAIL"},
    ]
    cases = import_from_logs(
        records, input_field="input_text", expected_field="label", tags=[]
    )
    assert all(isinstance(c, TestCase) for c in cases)


def test_import_from_logs_count():
    records = [{"input_text": f"query {i}"} for i in range(5)]
    cases = import_from_logs(
        records, input_field="input_text", expected_field=None, tags=[]
    )
    assert len(cases) == 5


def test_import_from_logs_ids_are_auto_generated_and_unique():
    records = [{"input_text": f"query {i}"} for i in range(4)]
    cases = import_from_logs(
        records, input_field="input_text", expected_field=None, tags=[]
    )
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "auto-generated ids must be unique"


def test_import_from_logs_tags_applied():
    records = [{"input_text": "hello"}]
    cases = import_from_logs(
        records, input_field="input_text", expected_field=None, tags=["en", "greeting"]
    )
    assert "en" in cases[0].tags
    assert "greeting" in cases[0].tags


def test_import_from_logs_expected_field_mapped():
    records = [{"input_text": "hello", "answer": "urgent"}]
    cases = import_from_logs(
        records, input_field="input_text", expected_field="answer", tags=[]
    )
    assert cases[0].expected == "urgent"


def test_import_from_logs_no_expected_field_is_none():
    records = [{"input_text": "hello"}]
    cases = import_from_logs(
        records, input_field="input_text", expected_field=None, tags=[]
    )
    assert cases[0].expected is None


def test_import_from_logs_empty_records():
    cases = import_from_logs([], input_field="input_text", expected_field=None, tags=[])
    assert cases == []


def test_import_from_logs_input_stored_in_input_dict():
    records = [{"input_text": "what time is it"}]
    cases = import_from_logs(
        records, input_field="input_text", expected_field=None, tags=[]
    )
    # The input field value must appear somewhere in case.input
    assert "what time is it" in cases[0].input.values()
