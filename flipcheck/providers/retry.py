"""Retry helper for provider calls.

Covers SPEC section 6 (retry on 429 and 5xx). All RuntimeError exceptions
are treated as retryable (they cover 429 and 5xx codes raised by LiteLLM).
Any other exception type is re-raised immediately without retry.

After max_attempts retryable failures the function returns a Sample with
the error field set rather than raising, so the runner can continue.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Coroutine
from typing import Any

from flipcheck.types import Sample


def _is_retryable(exc: BaseException) -> bool:
    """Return True when exc is a RuntimeError that warrants a retry."""
    return isinstance(exc, RuntimeError)


async def with_retry(
    coro_fn: Callable[[], Coroutine[Any, Any, Any]],
    max_attempts: int = 3,
) -> Any:
    """Retry a coroutine function on retryable (RuntimeError) failures.

    Args:
        coro_fn: A zero-argument callable that returns a coroutine each time
                 it is called. It is called once per attempt.
        max_attempts: Maximum number of attempts before giving up.

    Returns:
        The value returned by coro_fn on success, or a Sample with error set
        if all attempts are exhausted.

    Raises:
        Any non-RuntimeError exception from coro_fn immediately, without retry.
    """
    last_exc: RuntimeError | None = None

    for attempt in range(max_attempts):
        try:
            return await coro_fn()
        except RuntimeError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = 0.1 * (2**attempt) + random.uniform(0, 0.1)
                await asyncio.sleep(delay)
        except Exception:
            raise

    return Sample(
        case_id="",
        version_id="",
        sample_idx=0,
        output=None,
        error=str(last_exc),
    )
