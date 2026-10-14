"""Deterministic mock provider for tests and local development.

Covers SPEC section 6 (mock target). Outputs are seeded from the version ID,
case ID and sample index so each combination always yields the same result.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from flipcheck.types import Sample, TestCase, Version


class MockProvider:
    """A deterministic provider that returns PASS or FAIL based on configured probabilities.

    The seed is derived from the provider seed, version_id, case_id and sample_idx
    using MD5 so the mapping is stable across processes.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise from a mock config dict.

        Expected keys:
            pass_prob: mapping of case_id (or "*") to a float in [0, 1].
            seed: integer seed for reproducibility.
        """
        self._pass_prob: dict[str, float] = config.get("pass_prob", {"*": 0.8})
        self._seed: int = int(config.get("seed", 0))

    async def generate(self, version: Version, case: TestCase, sample_idx: int) -> Sample:
        """Return a Sample with output PASS or FAIL.

        The result is deterministic given the same version_id, case_id and sample_idx.
        """
        key = f"{self._seed}:{version.version_id}:{case.id}:{sample_idx}".encode()
        rng_seed = int(hashlib.md5(key).hexdigest(), 16) % (2**32)
        rng = random.Random(rng_seed)

        prob = self._pass_prob.get(case.id, self._pass_prob.get("*", 0.8))
        output = "PASS" if rng.random() < prob else "FAIL"

        return Sample(
            case_id=case.id,
            version_id=version.version_id,
            sample_idx=sample_idx,
            output=output,
            error=None,
        )
