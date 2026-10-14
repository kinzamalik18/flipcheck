"""LiteLLM-backed provider. Implemented in a later phase."""

from __future__ import annotations

from flipcheck.types import Sample, TargetConfig, TestCase, Version


class LLMProvider:
    """Provider that calls an LLM via LiteLLM."""

    def __init__(self, config: TargetConfig) -> None:
        self._config = config

    async def generate(self, version: Version, case: TestCase, sample_idx: int) -> Sample:
        """Generate one sample using LiteLLM."""
        raise NotImplementedError("LLMProvider is not implemented yet.")
