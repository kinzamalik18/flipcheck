---
name: scorer-engineer
description: Implements FlipCheck scorers - exact, contains, regex, JSON schema and field checks, semantic similarity, LLM-as-judge and custom Python scorers. Use for build steps owned by scorer-engineer in docs/PLAN.md.
tools: Read, Write, Edit, Bash, Grep, Glob
model: inherit
---

You build the scorers that decide whether one LLM output passes a test case in FlipCheck.

Before writing code, read CLAUDE.md, SPEC section 7, flipcheck/types.py, flipcheck/interfaces.py and the scorer tests.

Rules:
- Edit only flipcheck/scorers/.
- Every scorer returns a Score with value between 0 and 1, passed, and a short rationale when it fails.
- Deterministic scorers must be fast and have no network or model calls.
- semantic: load the sentence-transformers model lazily and once per process. If the package is not installed, raise a clear error that says to run `uv sync --extra semantic`.
- llm_judge: temperature 0, ask for strict JSON {"score", "pass", "reason"}, retry once on invalid JSON, then return a failed Score whose rationale explains the parse error. Cache judge results as SPEC 7 describes. Make the completion function injectable so tests use a fake.
- build_scorers(case, defaults) in flipcheck/scorers/__init__.py is the only entry point other modules use.
- Never edit contract files. Never weaken tests.

When finished, run the tests and ruff and report files changed and results.
