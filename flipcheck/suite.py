"""Suite loading and log import utilities for FlipCheck.

Covers SPEC section 5.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from flipcheck.types import TestCase

VALID_SCORER_TYPES: frozenset[str] = frozenset(
    {
        "exact",
        "contains",
        "regex",
        "json_schema",
        "json_field",
        "semantic",
        "llm_judge",
        "python",
    }
)


def _validate_cases(cases: list[TestCase]) -> None:
    """Check for duplicate IDs and invalid scorer types.

    Args:
        cases: The list of parsed TestCase objects to validate.

    Raises:
        ValueError: If any id is duplicated or any scorer type is unknown.
    """
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(
                f"Duplicate id found in suite: '{case.id}'. Each test case must have a unique id."
            )
        seen.add(case.id)
        for scorer in case.scorers:
            if scorer.type not in VALID_SCORER_TYPES:
                raise ValueError(
                    f"Unknown scorer type '{scorer.type}' on case '{case.id}'. "
                    f"Valid types are: {sorted(VALID_SCORER_TYPES)}."
                )


def _load_jsonl(path: Path) -> list[TestCase]:
    """Parse a JSONL file into TestCase objects, one per non-empty line."""
    cases: list[TestCase] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
        try:
            cases.append(TestCase.model_validate(data))
        except ValidationError as exc:
            raise ValueError(f"Invalid test case on line {lineno}: {exc}") from exc
    return cases


def _load_csv(path: Path) -> list[TestCase]:
    """Parse a CSV file into TestCase objects.

    Expected columns: id, input_query, expected, tags, critical.
    """
    cases: list[TestCase] = []
    with open(str(path), newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for rownum, row in enumerate(reader, start=2):
            tags_raw = row.get("tags", "")
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

            critical_raw = row.get("critical", "false").strip().lower()
            critical = critical_raw == "true"

            data = {
                "id": row["id"],
                "input": {"query": row["input_query"]},
                "expected": row.get("expected") or None,
                "tags": tags,
                "critical": critical,
            }
            try:
                cases.append(TestCase.model_validate(data))
            except ValidationError as exc:
                raise ValueError(f"Invalid test case on CSV row {rownum}: {exc}") from exc
    return cases


def _load_yaml(path: Path) -> list[TestCase]:
    """Parse a YAML file containing a list of test case dicts."""
    raw = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in suite file: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("YAML suite file must contain a list of test cases at the top level.")

    cases: list[TestCase] = []
    for idx, item in enumerate(data):
        try:
            cases.append(TestCase.model_validate(item))
        except ValidationError as exc:
            raise ValueError(f"Invalid test case at YAML index {idx}: {exc}") from exc
    return cases


def load_suite(path: str) -> list[TestCase]:
    """Load a test suite from a JSONL, CSV or YAML file.

    Format is detected by file extension: .jsonl, .csv, .yaml or .yml.

    Args:
        path: Path to the suite file.

    Returns:
        A list of validated TestCase objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any id is duplicated, any scorer type is unknown,
            required fields are missing, or the format is unsupported.
    """
    suite_path = Path(path)
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite file not found: {path}")

    suffix = suite_path.suffix.lower()
    if suffix == ".jsonl":
        cases = _load_jsonl(suite_path)
    elif suffix == ".csv":
        cases = _load_csv(suite_path)
    elif suffix in {".yaml", ".yml"}:
        cases = _load_yaml(suite_path)
    else:
        raise ValueError(
            f"Unsupported suite file format '{suffix}'. Use .jsonl, .csv, .yaml or .yml."
        )

    _validate_cases(cases)
    return cases


def import_from_logs(
    records: list[dict],
    input_field: str,
    expected_field: str | None,
    tags: list[str],
) -> list[TestCase]:
    """Convert a list of log record dicts into TestCase objects.

    Each case receives an auto-generated id in the form 'imported_NNNN'
    (1-based, zero-padded to 4 digits).

    Args:
        records: List of log record dicts.
        input_field: Key in each record to use as the input value.
        expected_field: Key in each record to use as the expected value,
            or None to leave expected unset.
        tags: List of tag strings to attach to every imported case.

    Returns:
        A list of TestCase objects, empty if records is empty.
    """
    cases: list[TestCase] = []
    for i, record in enumerate(records):
        expected = None
        if expected_field is not None and expected_field in record:
            expected = record[expected_field]

        cases.append(
            TestCase(
                id=f"imported_{i + 1:04d}",
                input={input_field: record[input_field]},
                expected=expected,
                tags=list(tags),
                critical=False,
            )
        )
    return cases
