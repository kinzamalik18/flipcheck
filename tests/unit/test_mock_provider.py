"""Tests for flipcheck/providers/mock.py: deterministic mock provider.

Covers SPEC section 6 (mock target).
"""

from __future__ import annotations

import pytest

import flipcheck.providers.mock as _mock_mod
from flipcheck.types import Sample, TargetConfig, TestCase, Version

if not hasattr(_mock_mod, "MockProvider"):
    pytest.skip("flipcheck.providers.mock.MockProvider not implemented yet", allow_module_level=True)

MockProvider = _mock_mod.MockProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_version(pass_prob: dict, seed: int = 1, version_id: str = "ver000000001") -> Version:
    cfg = TargetConfig(type="mock", mock={"pass_prob": pass_prob, "seed": seed})
    return Version(name="baseline", config=cfg, version_id=version_id)


def _make_case(case_id: str = "t001", expected: str = "PASS") -> TestCase:
    return TestCase(id=case_id, input={"query": "hello"}, expected=expected)


# ---------------------------------------------------------------------------
# Basic output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_provider_returns_sample():
    provider = MockProvider({"pass_prob": {"*": 0.8}, "seed": 1})
    version = _make_version({"*": 0.8})
    case = _make_case()
    sample = await provider.generate(version, case, sample_idx=0)
    assert isinstance(sample, Sample)


@pytest.mark.asyncio
async def test_mock_provider_output_is_pass_or_fail():
    provider = MockProvider({"pass_prob": {"*": 0.8}, "seed": 1})
    version = _make_version({"*": 0.8})
    case = _make_case()
    sample = await provider.generate(version, case, sample_idx=0)
    assert sample.output in ("PASS", "FAIL")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_seed_version_case_idx_always_same_output():
    provider = MockProvider({"pass_prob": {"*": 0.8}, "seed": 42})
    version = _make_version({"*": 0.8}, seed=42, version_id="ver000000042")
    case = _make_case(case_id="t001")
    outputs = [
        (await provider.generate(version, case, sample_idx=0)).output for _ in range(5)
    ]
    assert len(set(outputs)) == 1, "same seed/version/case/idx must always give the same output"


@pytest.mark.asyncio
async def test_different_sample_idx_produces_variety():
    """Across many sample indices, not all outputs should be identical.

    With pass_prob=0.5 and 50 draws the probability that all are identical
    is (0.5)^49 which is negligible.
    """
    provider = MockProvider({"pass_prob": {"*": 0.5}, "seed": 7})
    version = _make_version({"*": 0.5}, seed=7)
    case = _make_case()
    outputs = {(await provider.generate(version, case, sample_idx=i)).output for i in range(50)}
    assert len(outputs) > 1, "different sample_idx values should produce both PASS and FAIL"


# ---------------------------------------------------------------------------
# Per-case pass_prob
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pass_prob_one_always_pass():
    provider = MockProvider({"pass_prob": {"t001": 1.0, "*": 0.0}, "seed": 1})
    version = _make_version({"t001": 1.0, "*": 0.0})
    case = _make_case(case_id="t001")
    for idx in range(10):
        sample = await provider.generate(version, case, sample_idx=idx)
        assert sample.output == "PASS", f"sample_idx={idx} should be PASS"


@pytest.mark.asyncio
async def test_pass_prob_zero_wildcard_always_fail_other_cases():
    provider = MockProvider({"pass_prob": {"t001": 1.0, "*": 0.0}, "seed": 1})
    version = _make_version({"t001": 1.0, "*": 0.0})
    case = _make_case(case_id="t999")
    for idx in range(10):
        sample = await provider.generate(version, case, sample_idx=idx)
        assert sample.output == "FAIL", f"sample_idx={idx} should be FAIL for wildcard 0.0"


# ---------------------------------------------------------------------------
# Sample fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_has_correct_case_id():
    provider = MockProvider({"pass_prob": {"*": 0.8}, "seed": 1})
    version = _make_version({"*": 0.8}, version_id="ver000000001")
    case = _make_case(case_id="mycase")
    sample = await provider.generate(version, case, sample_idx=0)
    assert sample.case_id == "mycase"


@pytest.mark.asyncio
async def test_sample_has_correct_version_id():
    provider = MockProvider({"pass_prob": {"*": 0.8}, "seed": 1})
    version = _make_version({"*": 0.8}, version_id="ver000000001")
    case = _make_case()
    sample = await provider.generate(version, case, sample_idx=0)
    assert sample.version_id == "ver000000001"


@pytest.mark.asyncio
async def test_sample_has_correct_sample_idx():
    provider = MockProvider({"pass_prob": {"*": 0.8}, "seed": 1})
    version = _make_version({"*": 0.8})
    case = _make_case()
    sample = await provider.generate(version, case, sample_idx=3)
    assert sample.sample_idx == 3


@pytest.mark.asyncio
async def test_sample_error_is_none_on_normal_call():
    provider = MockProvider({"pass_prob": {"*": 0.8}, "seed": 1})
    version = _make_version({"*": 0.8})
    case = _make_case()
    sample = await provider.generate(version, case, sample_idx=0)
    assert sample.error is None


# ---------------------------------------------------------------------------
# Async interface
# ---------------------------------------------------------------------------


def test_generate_is_awaitable():
    """generate() must return a coroutine (i.e., be an async method)."""
    import inspect

    provider = MockProvider({"pass_prob": {"*": 0.8}, "seed": 1})
    version = _make_version({"*": 0.8})
    case = _make_case()
    coro = provider.generate(version, case, sample_idx=0)
    assert inspect.isawaitable(coro), "generate() must be async"
    # Clean up the unawaited coroutine to avoid warnings
    coro.close()
