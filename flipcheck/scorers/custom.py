"""Custom Python callable scorer."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

from flipcheck.types import Score, TestCase


class PythonScorer:
    """Score output with a user-supplied Python callable."""

    name = "python"

    def __init__(
        self,
        args: dict | None = None,
        fn: Callable[..., Any] | None = None,
    ) -> None:
        if args is None:
            args = {}

        if fn is not None:
            self._fn: Callable[..., Any] = fn
        elif "fn" in args:
            candidate = args["fn"]
            if callable(candidate):
                self._fn = candidate
            else:
                raise TypeError("args['fn'] must be callable.")
        elif "callable" in args:
            self._fn = _load_callable(args["callable"])
        else:
            raise ValueError(
                "PythonScorer requires a callable via 'fn' keyword argument, "
                "args['fn'], or args['callable'] as 'module:function'."
            )

    async def score(self, case: TestCase, output: str) -> Score:
        """Call the user function and convert the result to a Score."""
        result = self._fn(case, output)

        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, Score):
            return result

        passed = bool(result)
        return Score(
            scorer=self.name,
            value=1.0 if passed else 0.0,
            passed=passed,
        )


def _load_callable(path: str) -> Callable[..., Any]:
    """Import a callable from a 'module:function' dotted path string."""
    if ":" not in path:
        raise ValueError(
            f"callable path {path!r} must be in 'module:function' format."
        )
    module_path, func_name = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name, None)
    if fn is None:
        raise ImportError(
            f"Function {func_name!r} not found in module {module_path!r}."
        )
    return fn
