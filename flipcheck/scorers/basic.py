"""Basic scorers: exact, contains, regex."""

from __future__ import annotations

import re

from flipcheck.types import Score, TestCase


class ExactScorer:
    """Compare output to expected, optionally case-insensitive and trimmed."""

    name = "exact"

    def __init__(self, args: dict | None = None) -> None:
        if args is None:
            args = {}
        self.field: str | None = args.get("field")
        self.normalize: bool = args.get("normalize", True)

    async def score(self, case: TestCase, output: str) -> Score:
        """Return Score(value=1.0) when output matches expected, else 0.0."""
        import json

        expected = case.expected
        actual = output

        if self.field is not None:
            try:
                parsed = json.loads(output)
            except (json.JSONDecodeError, ValueError):
                return Score(
                    scorer=self.name,
                    value=0.0,
                    passed=False,
                    rationale="Output is not valid JSON.",
                )
            actual = parsed.get(self.field)
            if actual is None and self.field not in parsed:
                return Score(
                    scorer=self.name,
                    value=0.0,
                    passed=False,
                    rationale=f"Field '{self.field}' not found in output.",
                )
            if isinstance(expected, dict):
                expected = expected.get(self.field)

        actual_str = str(actual) if actual is not None else ""
        expected_str = str(expected) if expected is not None else ""

        if self.normalize:
            actual_str = actual_str.strip().lower()
            expected_str = expected_str.strip().lower()

        passed = actual_str == expected_str
        return Score(
            scorer=self.name,
            value=1.0 if passed else 0.0,
            passed=passed,
            rationale=None if passed else f"Expected {expected_str!r}, got {actual_str!r}.",
        )


class ContainsScorer:
    """Check that output contains one or all of a list of substrings."""

    name = "contains"

    def __init__(self, args: dict | None = None) -> None:
        if args is None:
            args = {}
        self.values: list[str] = args.get("values", [])
        self.mode: str = args.get("mode", "any")

    async def score(self, case: TestCase, output: str) -> Score:
        """Return Score based on whether output contains required substrings."""
        if self.mode == "all":
            passed = all(v in output for v in self.values)
        else:
            passed = any(v in output for v in self.values)

        if not self.values and self.mode == "all":
            passed = True

        missing = [v for v in self.values if v not in output]
        rationale = None
        if not passed:
            rationale = f"Missing values: {missing!r}."

        return Score(
            scorer=self.name,
            value=1.0 if passed else 0.0,
            passed=passed,
            rationale=rationale,
        )


class RegexScorer:
    """Check that output matches (or does not match) a regex pattern."""

    name = "regex"

    def __init__(self, args: dict | None = None) -> None:
        if args is None:
            args = {}
        self.pattern: str = args.get("pattern", "")
        self.must_match: bool = args.get("must_match", True)

    async def score(self, case: TestCase, output: str) -> Score:
        """Return Score based on regex match against output."""
        matched = bool(re.search(self.pattern, output))
        passed = matched if self.must_match else not matched

        rationale = None
        if not passed:
            if self.must_match:
                rationale = f"Pattern {self.pattern!r} did not match output."
            else:
                rationale = f"Pattern {self.pattern!r} matched output but must not."

        return Score(
            scorer=self.name,
            value=1.0 if passed else 0.0,
            passed=passed,
            rationale=rationale,
        )
