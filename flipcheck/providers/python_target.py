"""Python callable provider. Implemented in a later phase."""

from __future__ import annotations

from flipcheck.types import Sample, TargetConfig, TestCase, Version


class PythonProvider:
    """Provider that calls a Python function by dotted import path."""

    def __init__(self, config: TargetConfig) -> None:
        self._config = config

    async def generate(self, version: Version, case: TestCase, sample_idx: int) -> Sample:
        """Generate one sample by calling the configured Python callable."""
        raise NotImplementedError("PythonProvider is not implemented yet.")
