"""HTTP POST provider using httpx. Implemented in a later phase."""

from __future__ import annotations

from flipcheck.types import Sample, TargetConfig, TestCase, Version


class HTTPProvider:
    """Provider that calls an HTTP endpoint."""

    def __init__(self, config: TargetConfig) -> None:
        self._config = config

    async def generate(self, version: Version, case: TestCase, sample_idx: int) -> Sample:
        """Generate one sample by POSTing to the configured URL."""
        raise NotImplementedError("HTTPProvider is not implemented yet.")
