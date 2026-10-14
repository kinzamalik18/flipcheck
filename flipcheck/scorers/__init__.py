"""Scorer factory for FlipCheck.

The only entry point other modules should use is build_scorers().
"""

from __future__ import annotations

from flipcheck.interfaces import Scorer
from flipcheck.types import ScorerConfig, TestCase


def build_scorers(case: TestCase, default_scorers: list[ScorerConfig]) -> list[Scorer]:
    """Build scorer instances for a test case.

    Uses case.scorers if non-empty, otherwise falls back to default_scorers.
    """
    configs = case.scorers if case.scorers else default_scorers
    return [_make_scorer(cfg) for cfg in configs]


def _make_scorer(cfg: ScorerConfig) -> Scorer:
    """Instantiate a scorer from its config."""
    if cfg.type == "exact":
        from flipcheck.scorers.basic import ExactScorer

        return ExactScorer(cfg.args)
    if cfg.type == "contains":
        from flipcheck.scorers.basic import ContainsScorer

        return ContainsScorer(cfg.args)
    if cfg.type == "regex":
        from flipcheck.scorers.basic import RegexScorer

        return RegexScorer(cfg.args)
    if cfg.type == "json_schema":
        from flipcheck.scorers.structured import JsonSchemaScorer

        return JsonSchemaScorer(cfg.args)
    if cfg.type == "json_field":
        from flipcheck.scorers.structured import JsonFieldScorer

        return JsonFieldScorer(cfg.args)
    if cfg.type == "llm_judge":
        from flipcheck.scorers.judge import LLMJudgeScorer

        return LLMJudgeScorer(cfg.args)
    if cfg.type == "python":
        from flipcheck.scorers.custom import PythonScorer

        return PythonScorer(cfg.args)
    if cfg.type == "semantic":
        from flipcheck.scorers.semantic import SemanticScorer

        return SemanticScorer(cfg.args)
    raise ValueError(f"Unknown scorer type: {cfg.type}")
