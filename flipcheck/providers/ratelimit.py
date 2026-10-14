"""Token-bucket rate limiter for provider calls.

Covers SPEC section 6 (rate limiting). Uses asyncio.Semaphore with a
call_later release to enforce a sliding-window requests-per-second limit.
"""

from __future__ import annotations

import asyncio
from types import TracebackType


class TokenBucketRateLimiter:
    """Async context manager that limits throughput to rate requests per second.

    If rate is 0 the limiter is a no-op: it never blocks.

    Usage:
        limiter = TokenBucketRateLimiter(rate=10)
        async with limiter:
            ...  # at most 10 of these run per second
    """

    def __init__(self, rate: int) -> None:
        """Create a rate limiter.

        Args:
            rate: Maximum requests per second. 0 means no limit.
        """
        self._rate = rate
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(rate) if rate > 0 else None
        )

    async def __aenter__(self) -> TokenBucketRateLimiter:
        """Acquire a token, blocking if the per-second budget is exhausted."""
        if self._semaphore is None:
            return self
        await self._semaphore.acquire()
        loop = asyncio.get_running_loop()
        loop.call_later(1.0, self._semaphore.release)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release is handled automatically via call_later; nothing to do here."""
