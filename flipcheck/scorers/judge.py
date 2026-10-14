"""LLM judge scorer."""

from __future__ import annotations

import json
from collections.abc import Callable, Coroutine
from typing import Any

from flipcheck.types import Score, TestCase


class LLMJudgeScorer:
    """Score output by asking an LLM judge to return a JSON evaluation."""

    name = "llm_judge"

    def __init__(
        self,
        args: dict | None = None,
        judge_config: Any = None,
        judge_fn: Callable[..., Coroutine[Any, Any, str]] | None = None,
    ) -> None:
        if args is None:
            args = {}
        self.rubric: str = args.get("rubric", "")
        self.pass_threshold: float = args.get("pass_threshold", 0.5)
        self._judge_fn = judge_fn
        self._judge_config = judge_config

    async def score(self, case: TestCase, output: str) -> Score:
        """Call the judge function, parse JSON, retry once on invalid JSON."""
        raw = await self._call_judge(case, output)
        result = _parse_judge_response(raw)
        if result is None:
            raw2 = await self._call_judge(case, output)
            result = _parse_judge_response(raw2)
            if result is None:
                return Score(
                    scorer=self.name,
                    value=0.0,
                    passed=False,
                    rationale=f"Judge returned invalid JSON after retry: {raw2!r}",
                )

        return Score(
            scorer=self.name,
            value=float(result["score"]),
            passed=bool(result["pass"]),
            rationale=result.get("reason"),
        )

    async def _call_judge(self, case: TestCase, output: str) -> str:
        """Invoke the judge function (injected or real)."""
        if self._judge_fn is not None:
            return await self._judge_fn(case, output, self.rubric)
        raise NotImplementedError(
            "No judge_fn provided. Configure a judge model in flipcheck.yaml."
        )


def _parse_judge_response(raw: str) -> dict | None:
    """Parse the judge JSON response. Returns None on any parse failure."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if "score" not in data or "pass" not in data:
        return None
    return data
