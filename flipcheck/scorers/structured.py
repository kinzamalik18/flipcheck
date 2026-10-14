"""Structured output scorers: json_schema and json_field."""

from __future__ import annotations

import json

from flipcheck.types import Score, TestCase


class JsonSchemaScorer:
    """Validate output as JSON against a JSON Schema."""

    name = "json_schema"

    def __init__(self, args: dict | None = None) -> None:
        if args is None:
            args = {}
        self.schema: dict = args.get("schema", {})

    async def score(self, case: TestCase, output: str) -> Score:
        """Return Score(passed=True) when output is valid JSON matching the schema."""
        try:
            import jsonschema
        except ImportError as exc:
            raise ImportError(
                "jsonschema is required for json_schema scorer. Run: uv sync"
            ) from exc

        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return Score(
                scorer=self.name,
                value=0.0,
                passed=False,
                rationale="Output is not valid JSON.",
            )

        try:
            jsonschema.validate(instance=parsed, schema=self.schema)
        except jsonschema.ValidationError as exc:
            return Score(
                scorer=self.name,
                value=0.0,
                passed=False,
                rationale=exc.message,
            )

        return Score(scorer=self.name, value=1.0, passed=True)


class JsonFieldScorer:
    """Parse output as JSON and compare a single field to case.expected."""

    name = "json_field"

    def __init__(self, args: dict | None = None) -> None:
        if args is None:
            args = {}
        self.field: str = args.get("field", "")
        self.normalize: bool = args.get("normalize", True)

    async def score(self, case: TestCase, output: str) -> Score:
        """Return Score based on whether output[field] matches case.expected."""
        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return Score(
                scorer=self.name,
                value=0.0,
                passed=False,
                rationale="Output is not valid JSON.",
            )

        if self.field not in parsed:
            return Score(
                scorer=self.name,
                value=0.0,
                passed=False,
                rationale=f"Field '{self.field}' not found in output.",
            )

        actual = str(parsed[self.field]) if parsed[self.field] is not None else ""
        expected = str(case.expected) if case.expected is not None else ""

        if self.normalize:
            actual = actual.strip().lower()
            expected = expected.strip().lower()

        passed = actual == expected
        return Score(
            scorer=self.name,
            value=1.0 if passed else 0.0,
            passed=passed,
            rationale=None if passed else f"Expected {expected!r}, got {actual!r}.",
        )
