"""Semantic similarity scorer using sentence-transformers."""

from __future__ import annotations

from flipcheck.types import Score, TestCase

_model_cache: dict[str, object] = {}


def _load_model(model_name: str) -> object:
    """Load a sentence-transformers model lazily, once per process."""
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: uv sync --extra semantic"
        ) from exc
    model = SentenceTransformer(model_name)
    _model_cache[model_name] = model
    return model


class SemanticScorer:
    """Score output by cosine similarity with case.expected."""

    name = "semantic"

    def __init__(self, args: dict | None = None) -> None:
        if args is None:
            args = {}
        self.threshold: float = args.get("threshold", 0.8)
        self.model_name: str = args.get("model", "all-MiniLM-L6-v2")

    async def score(self, case: TestCase, output: str) -> Score:
        """Return Score based on cosine similarity between output and expected."""
        import numpy as np

        model = _load_model(self.model_name)
        expected = str(case.expected) if case.expected is not None else ""
        embeddings = model.encode([expected, output], convert_to_numpy=True)  # type: ignore[attr-defined]
        a, b = embeddings[0], embeddings[1]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        similarity = 0.0 if norm_a == 0 or norm_b == 0 else float(np.dot(a, b) / (norm_a * norm_b))

        passed = similarity >= self.threshold
        return Score(
            scorer=self.name,
            value=float(similarity),
            passed=passed,
            rationale=None if passed else f"Similarity {similarity:.3f} below threshold {self.threshold}.",
        )
