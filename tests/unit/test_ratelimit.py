"""Tests for flipcheck/providers/ratelimit.py and flipcheck/providers/retry.py.

Covers SPEC section 6 (rate limiting and retry behavior).
"""

from __future__ import annotations

import asyncio
import time

import pytest

import flipcheck.providers.ratelimit as _ratelimit_mod
import flipcheck.providers.retry as _retry_mod

if not hasattr(_ratelimit_mod, "TokenBucketRateLimiter"):
    pytest.skip(
        "flipcheck.providers.ratelimit.TokenBucketRateLimiter not implemented yet",
        allow_module_level=True,
    )
if not hasattr(_retry_mod, "with_retry"):
    pytest.skip(
        "flipcheck.providers.retry.with_retry not implemented yet",
        allow_module_level=True,
    )

TokenBucketRateLimiter = _ratelimit_mod.TokenBucketRateLimiter
with_retry = _retry_mod.with_retry


# ---------------------------------------------------------------------------
# TokenBucketRateLimiter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiter_allows_up_to_limit_immediately():
    """A limiter with limit=2 must not block the first two requests."""
    limiter = TokenBucketRateLimiter(rate=2)
    start = time.monotonic()
    async with limiter:
        pass
    async with limiter:
        pass
    elapsed = time.monotonic() - start
    # Both should complete well under 1 second (no blocking)
    assert elapsed < 1.0, f"First two requests took {elapsed:.2f}s, expected < 1s"


@pytest.mark.asyncio
async def test_rate_limiter_blocks_third_request_briefly():
    """A limiter with limit=2 per second must delay a third concurrent request."""
    limiter = TokenBucketRateLimiter(rate=2)

    async def single_request():
        async with limiter:
            pass

    start = time.monotonic()
    # Fire 3 requests concurrently
    await asyncio.gather(single_request(), single_request(), single_request())
    elapsed = time.monotonic() - start
    # With only 2 tokens per second the third must wait ~0.5s
    assert elapsed >= 0.3, f"Rate limiter did not block: elapsed {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_rate_limiter_zero_means_no_limit():
    """limit=0 must allow unlimited requests without blocking."""
    limiter = TokenBucketRateLimiter(rate=0)
    start = time.monotonic()
    for _ in range(20):
        async with limiter:
            pass
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"Unlimited limiter took {elapsed:.2f}s for 20 requests"


# ---------------------------------------------------------------------------
# with_retry: success after transient failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_after_two_failures():
    """A coroutine that fails twice then succeeds should return successfully."""
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("transient 503")
        return "ok"

    result = await with_retry(flaky, max_attempts=3)
    assert result == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_call_count_matches_attempts():
    """Verify the exact number of attempts before success."""
    call_count = 0

    async def succeed_on_third():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("retry me")
        return "done"

    await with_retry(succeed_on_third, max_attempts=5)
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_raises_on_non_retryable_error():
    """ValueError should not be retried (it is not a 429/5xx)."""
    call_count = 0

    async def always_value_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await with_retry(always_value_error, max_attempts=3)

    # Should have been called only once (no retry on ValueError)
    assert call_count == 1


@pytest.mark.asyncio
async def test_retry_immediate_success_no_retry():
    """A function that succeeds on the first try should not retry."""
    call_count = 0

    async def always_ok():
        nonlocal call_count
        call_count += 1
        return "success"

    result = await with_retry(always_ok, max_attempts=3)
    assert result == "success"
    assert call_count == 1


# ---------------------------------------------------------------------------
# with_retry: error set after exhausting attempts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_exhausted_does_not_raise_returns_error_sample():
    """After max_attempts retryable failures, with_retry must return a Sample
    with error set, not raise an exception.

    SPEC section 6: 'After that the sample gets error set, not an exception raised.'
    This test uses a retryable error (simulated 429/5xx).
    """
    from flipcheck.types import Sample

    async def always_rate_limit():
        raise RuntimeError("429 rate limited")

    result = await with_retry(always_rate_limit, max_attempts=3)
    assert isinstance(result, Sample), "exhausted retries must return a Sample, not raise"
    assert result.error is not None, "error field must be set on the returned Sample"
    assert result.output is None


@pytest.mark.asyncio
async def test_retry_three_failures_gives_error_in_sample():
    """Specifically three 5xx failures must yield a Sample with error, per the spec."""
    from flipcheck.types import Sample

    attempts = 0

    async def always_500():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("500 internal server error")

    result = await with_retry(always_500, max_attempts=3)
    assert isinstance(result, Sample)
    assert result.error is not None
    assert attempts == 3
