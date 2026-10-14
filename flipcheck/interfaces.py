"""Protocol interfaces for FlipCheck.

These define the contracts that providers, scorers and the store must satisfy.
This file is orchestrator-owned: subagents must not edit it.
"""

from __future__ import annotations

from typing import Protocol

from flipcheck.types import Sample, Score, TestCase, Version


class Provider(Protocol):
    """Protocol for LLM, HTTP, Python, and mock targets."""

    async def generate(self, version: Version, case: TestCase, sample_idx: int) -> Sample:
        """Generate one sample for a test case."""
        ...


class Scorer(Protocol):
    """Protocol for scorers."""

    name: str

    async def score(self, case: TestCase, output: str) -> Score:
        """Score a single output against a test case."""
        ...


class Store(Protocol):
    """Protocol for the storage layer. Only flipcheck/store/ writes SQL."""

    def save_sample(self, run_id: str, sample: Sample) -> None:
        """Persist a sample to the store."""
        ...

    def get_samples(self, run_id: str, case_id: str, version_id: str) -> list[Sample]:
        """Return all samples for a given run, case and version."""
        ...

    def save_item_results(self, run_id: str, results: list) -> None:
        """Persist the list of ItemResult objects for a run."""
        ...

    def list_runs(self) -> list[dict]:
        """Return a list of run metadata dicts, most recent first."""
        ...
