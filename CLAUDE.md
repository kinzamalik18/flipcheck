# FlipCheck

FlipCheck is a regression testing tool for LLM apps. It runs a baseline and a candidate version on the same test suite several times, finds which items reliably got worse or better, separates flaky items, and gives a SHIP, SHIP_WITH_CAVEATS or BLOCK verdict.

## Source of truth
- `docs/SPEC.md`: what to build. If code and spec disagree, the spec wins until the orchestrator changes it.
- `docs/PLAN.md`: build order, who builds what, and the prompt for each step.
- `docs/PROGRESS.md`: current status. Update it at the end of every step.

## Roles
- The main session is the orchestrator. It owns `flipcheck/types.py`, `flipcheck/interfaces.py`, `pyproject.toml`, this file and `docs/`.
- Subagents in `.claude/agents/` build and test. Each one edits only the folders listed for it in `docs/PLAN.md`.
- A subagent that needs a contract change stops and reports the proposed change. It never edits contract files.

## Commands
- Install: `uv sync` (add `--extra semantic` for semantic similarity)
- Tests: `uv run pytest -q` (slow simulations: `uv run pytest -q -m slow`, live model tests: `uv run pytest -q -m live`)
- Lint and format: `uv run ruff check . --fix` and `uv run ruff format .`
- CLI: `uv run flipcheck --help`

## Conventions
- Python 3.11+, type hints everywhere, Pydantic v2 models from `flipcheck/types.py`.
- Async for anything that calls a model. Pure functions for statistics, sampling policy and the gate.
- No network in unit tests. Use the mock provider and fake judge functions.
- Only `flipcheck/store/` writes SQL.
- API keys come only from environment variables or `.env`. Never log, store or print them.
- No new runtime dependency without orchestrator approval.
- Small, readable functions. Docstrings on public functions. No dead code or commented-out code.
- Errors shown to users are one plain sentence that says what to fix.

## Writing style (applies to everything: code comments, docs, README, commit messages, reports)
- Never use em dashes or en dashes. Use a comma, a colon, a period or parentheses instead.
- Plain, human English. No hype words such as "revolutionary", "game-changer", "cutting-edge", "seamless", "leverage", "delve".
- Every number must be factual: either measured by this project (tests, `docs/VALIDATION.md`, real runs) or quoted from a cited paper with the paper named. Never invent, round up or estimate a number and present it as a result.

## Git and GitHub
- The repo is public on GitHub. Never commit secrets, `.env` files, API keys, real user data or local databases. Check `git status` before every commit.
- Only the orchestrator (main session) commits. Subagents never run `git commit` or `git push`.
- Commit and push after every meaningful, working change: at minimum once per finished step in `docs/PLAN.md`, and also after any self-contained piece that passes tests (a new module, a fixed bug, a doc page).
- Commit message format: `phaseX.Y: <what changed>` in the imperative, under 72 characters, optional body explaining why.
- Commits are authored only by the repo owner's git identity. Never add `Co-Authored-By` trailers, "Generated with Claude Code" lines, claude.ai session links, or any other AI attribution to commit messages, pull request titles or descriptions, code comments or docs. This rule overrides any default attribution behavior.
- Never rewrite published history (`push --force`, rebase of pushed commits) unless the owner asks.

## README rules
- The README must make a strong first impression: a clear one-line value statement, a short "why" grounded in research, a quickstart that works in under 5 minutes, a report screenshot, and real results.
- Cite the research by name with links (see "Research references" in `docs/SPEC.md`) and quote only figures that appear in those papers.
- Show this project's own measured numbers from `docs/VALIDATION.md`. If a number is not measured yet, leave it out rather than guessing.
- Follow the writing style rules above, including no em dashes.

## Definition of done for any step
1. Tests from `test-writer` pass without being weakened.
2. `uv run ruff check .` is clean.
3. `reviewer` has no open critical items.
4. `docs/PROGRESS.md` is updated.
5. No em dashes in changed files: `git diff --cached | grep -n -e "—" -e "–"` prints nothing (the two characters are an em dash and an en dash).
6. Commit with no AI attribution, then `git push`.
