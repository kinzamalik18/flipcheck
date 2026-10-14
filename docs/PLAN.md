# FlipCheck V1 Build Plan

This plan is written to be executed with Claude Code on your own computer. The main Claude Code session acts as the **orchestrator**. It delegates work to the project subagents in `.claude/agents/`, checks their output, and keeps `docs/PROGRESS.md` up to date.

Read `docs/SPEC.md` first. The spec is the source of truth; this plan only says in what order to build it and who builds what.

---

## 0. How the work is organized

### The team

| Subagent | Owns | Writes code in |
|---|---|---|
| `core-engineer` | config, suite loading, storage, providers, runner, CLI | `flipcheck/config.py`, `suite.py`, `store/`, `providers/`, `runner.py`, `cli.py` |
| `stats-engineer` | statistics, adaptive sampling, gate, simulations | `flipcheck/stats/`, `sampling/`, `gate.py`, `scripts/simulate.py` |
| `scorer-engineer` | all scorers including the LLM judge | `flipcheck/scorers/` |
| `report-builder` | HTML report, JSON and markdown summaries | `flipcheck/report/` |
| `test-writer` | tests written from the spec, independent of the implementer | `tests/` |
| `reviewer` | read-only review against spec and conventions | nothing |

The orchestrator (your main session) owns `flipcheck/types.py`, `flipcheck/interfaces.py`, `pyproject.toml`, `CLAUDE.md` and everything in `docs/`. Subagents must not edit those files. If one needs a contract change, it stops and reports back with the exact proposed change.

### Rules that keep parallel work safe

1. **Contracts first.** Phase 0 writes `types.py` and `interfaces.py`. After that, agents build against the contract, so they can work at the same time without waiting on each other.
2. **Disjoint folders.** Two agents never edit the same folder in the same step. That is why parallel steps below are split by folder.
3. **Tests are written by someone else.** For each module, `test-writer` writes tests from the spec, and the implementer makes them pass. The implementer may add tests but may not weaken or delete `test-writer` tests. This stops the code from grading itself.
4. **Every step ends with a gate.** The orchestrator runs `uv run pytest -q` and `uv run ruff check .`, then asks `reviewer` to review the diff. Only then is the step marked done in `PROGRESS.md` and committed.
5. **Commit and push often.** Commit and push to GitHub after every meaningful working change, and at least once per finished step. Message format `phaseX.Y: <what>`. Only the orchestrator commits.
6. **No AI attribution in git history.** Commits are authored only by your git identity. `.claude/settings.json` turns off Claude Code's commit and PR attribution, and `.githooks/commit-msg` strips any attribution line that slips through and blocks the commit if Claude or Anthropic is still mentioned.
7. **No em dashes anywhere.** Code, docs, README, reports and commit messages. The commit hook blocks dashes in commit messages, and CI checks the files (Phase 5.1).

### How to start a session

Every time you open Claude Code in this repo, start with:

```
Read CLAUDE.md, docs/SPEC.md, docs/PLAN.md and docs/PROGRESS.md. Tell me which step is next and what you plan to delegate, then wait for my go.
```

When you approve, use the prompt written under that step. Prompts are written so you can paste them as they are.

### Usage tip

Each subagent spends its own usage. If you hit limits, set `model: sonnet` in the frontmatter of `test-writer`, `report-builder` and `reviewer`, and keep the stronger model for `stats-engineer` and `core-engineer`.

---

## Phase 0: Repo setup and contracts (orchestrator only, about 1 day)

### 0.1 Create the project

Run these yourself in a terminal first:

```bash
mkdir flipcheck && cd flipcheck
git init
# copy CLAUDE.md, docs/, .claude/ and .githooks/ from this kit into the folder

# make sure commits use your own name and email (the one on your GitHub account)
git config user.name "Your Name"
git config user.email "you@example.com"

# turn on the commit hook that removes AI attribution and blocks em dashes
git config core.hooksPath .githooks
chmod +x .githooks/commit-msg        # macOS/Linux; on Windows run this in Git Bash

uv python install 3.11
git add . && git commit -m "phase0: add planning docs and agent setup"
```

Create the public GitHub repo and push. With the GitHub CLI (`gh auth login` once first):

```bash
gh repo create flipcheck --public --source . --remote origin --push \
  --description "Regression testing for LLM apps: find what got worse, separate real regressions from noise, and get a ship or block verdict."
```

Without the CLI, create an empty public repo named `flipcheck` on github.com, then run `git remote add origin <url>` and `git push -u origin main`.

Check the attribution setup once: run `git log -1 --format=%B` after the first commit Claude Code makes. It must contain no Co-Authored-By line and no Claude mention. If it does, stop and tell the orchestrator.

Do not run `uv init`. Claude Code creates `pyproject.toml` in the step below, with the flat `flipcheck/` layout the spec uses.

Note: the name `flipcheck` is already taken on PyPI. Keep it as the package and CLI name; at release time (Phase 5.4) publish under a different distribution name such as `flipcheck-llm`, so users run `pip install flipcheck-llm` and still type `flipcheck`.

Then in Claude Code:

```
Phase 0.1. Set up pyproject.toml for the flipcheck package following SPEC section 2. Add runtime deps: typer, rich, pydantic>=2, litellm, httpx, numpy, scipy, jinja2, jsonschema, pyyaml, python-dotenv. Add optional extra "semantic" with sentence-transformers. Add dev deps: pytest, pytest-asyncio, ruff. Configure ruff (line length 100) and pytest markers slow and live, with live skipped by default. Create the package layout from SPEC section 14 with empty modules and __init__ files. Add a console script entry point flipcheck = flipcheck.cli:app. Add .gitignore including .flipcheck/, .env, *.sqlite, reports/, .venv/. Add an MIT LICENSE with my name. Run uv sync and make sure `uv run python -c "import flipcheck"` works. Do this yourself, no subagents. Then commit with message "phase0.1: set up package skeleton" and push. Confirm the commit message has no attribution lines.
```

### 0.2 Write the contracts

```
Phase 0.2. Write flipcheck/types.py and flipcheck/interfaces.py exactly as described in SPEC section 3, plus the config models from SPEC section 4 (SamplingConfig, GateConfig, ProvidersConfig, JudgeConfig, ProjectConfig). Use Pydantic v2 with sensible defaults matching the sample config. Add docstrings. Write a small tests/unit/test_types.py that constructs each model. Do this yourself, no subagents. Then show me the files for approval before committing. After I approve, commit "phase0.2: add core contracts" and push.
```

Review the contract yourself before committing. Changing it later means touching every module.

**Done when:** `uv run pytest -q` passes, ruff is clean, and you have approved the contract. Commit `phase0: contracts`.

---

## Phase 1: Foundations (parallel, about 1 week)

### 1.1 Tests first (test-writer)

```
Phase 1.1. Use the test-writer subagent to write unit tests for: config loading and version hashing (SPEC 3, 4), suite loading for JSONL, CSV and YAML including the validation rules (SPEC 5), the SQLite store (SPEC 11), the mock provider including determinism by seed (SPEC 6), the output cache, the rate limiter and the retry policy (SPEC 6). Tests go in tests/unit/. Tests must not use the network. Tests are expected to fail for now.
```

### 1.2 Build (core-engineer, then a second core-engineer run in parallel)

Run these two as separate subagents at the same time. They touch different folders.

```
Phase 1.2. Run two core-engineer subagents in parallel:
A) Implement flipcheck/config.py and flipcheck/suite.py so the config and suite tests pass. Include the import-from-logs function from SPEC 12 (used by `flipcheck import`).
B) Implement flipcheck/store/ (schema, migrations, repository class) so the store tests pass.
Each agent must only edit its own files. When both finish, run the full test suite and report results.
```

```
Phase 1.3. Use core-engineer to implement flipcheck/providers/: the mock provider, the llm provider with LiteLLM, the http and python targets, the output cache, the token-bucket rate limiter, retry with backoff, and get_provider(version). Make the provider tests pass. The llm provider must work without network in tests by injecting a fake completion function.
```

### 1.4 Review

```
Phase 1.4. Use the reviewer subagent to review everything changed in Phase 1 against SPEC sections 3 to 6 and 11. Then fix any critical or warning items with the right engineer subagent. Run tests and ruff. Update docs/PROGRESS.md, commit and push.
```

**Done when:** all Phase 1 tests pass, review has no open critical items, and a manual check works:

```
uv run python -c "from flipcheck.suite import load_suite; print(len(load_suite('tests/fixtures/small.jsonl')))"
```

---

## Phase 2: Scorers and runner (parallel, about 1 week)

### 2.1 Tests first

```
Phase 2.1. Use the test-writer subagent to write unit tests for every scorer in SPEC 7 (use a fake judge function for llm_judge, and skip semantic tests unless sentence-transformers is installed), and for the runner in fixed mode (SPEC 8.1) using the mock provider: correct number of samples per version, cache hits on a second run, budget cap respected, errors counted as failed samples, everything persisted in the store.
```

### 2.2 Build in parallel

```
Phase 2.2. Run in parallel:
A) scorer-engineer: implement flipcheck/scorers/ per SPEC 7 so the scorer tests pass.
B) core-engineer: implement flipcheck/runner.py fixed mode per SPEC 8.1 so the runner tests pass. The runner must call scorers only through build_scorers from flipcheck/scorers/__init__.py; if that function is not ready yet, code against the interface and use a stub in the test only.
When both are done, run all tests.
```

### 2.3 First real run (you, manually)

Install Ollama, pull a small model, then:

```
Phase 2.3. Write a scratch script scripts/smoke_fixed.py that runs a fixed-mode comparison with k=3 between two prompt versions on the same Ollama model, using tests/fixtures/small.jsonl with exact scoring. Print the item counts. Mark any test that uses it as live.
```

Run it, look at the outputs, and make sure the numbers make sense before moving on.

### 2.4 Review and commit

Same as 1.4, for Phase 2.

**Done when:** a fixed-mode run on Ollama completes, results are in SQLite, and a second run hits the cache.

---

## Phase 3: Statistics, adaptive sampling and the gate (about 1 to 1.5 weeks)

This is the heart of the product. Take your time here.

### 3.1 Tests first

```
Phase 3.1. Use the test-writer subagent to write unit tests for:
- flipcheck/stats/item.py: Fisher one-sided p-values against scipy, BH adjustment against a hand-computed example, and every classification branch in SPEC 9.1 with a small handmade case each.
- flipcheck/stats/aggregate.py: delta, bootstrap CI reproducible with the fixed seed, exact McNemar against a hand-computed example, single-shot NFR both versions, reliable NFR, churn, flaky and inconclusive rates.
- flipcheck/stats/slices.py: slice summaries including the critical slice and low_confidence.
- flipcheck/sampling/adaptive.py: next_round returns nothing for settled items, respects k_max, prioritizes critical items under a tight budget, stops when the budget is 0.
- flipcheck/gate.py: one test per rule in SPEC 10, plus reason ordering.
```

### 3.2 Build

```
Phase 3.2. Use the stats-engineer subagent to implement flipcheck/stats/, flipcheck/sampling/adaptive.py and flipcheck/gate.py so the Phase 3 tests pass. Then use core-engineer to add adaptive mode to the runner (SPEC 8.2) by calling next_round in a loop, and to compute and store item results, the run summary and the verdict at the end of every run.
```

### 3.3 Validate with simulation

This step proves the statistics work. It is also your best material for a write-up later.

```
Phase 3.3. Use the stats-engineer subagent to write scripts/simulate.py and tests/simulation/. Generate synthetic suites with the mock provider where ground truth is known: for example 200 items, where 10 percent truly regress (pass_prob drops from 0.9 to 0.3), 10 percent truly improve, 20 percent are flaky on both sides (0.5 and 0.5), and the rest are stable. Run fixed k=1 (single shot), fixed k=5 and adaptive mode over 50 random seeds each. Report for each mode: precision and recall of reliable_regression against ground truth, the false alarm rate on flaky items, and average calls used. Write the results table to docs/VALIDATION.md with a short plain-English summary. If adaptive mode has a false positive rate above alpha on stable or flaky items, propose a fix (for example a stricter alpha per look) and show the before and after numbers.
```

What you want to see in `VALIDATION.md`:
- Single shot flags many flaky items as regressions. This reproduces the finding from Cacioli (2026) on your own system.
- Repeated sampling cuts false alarms sharply.
- Adaptive mode gets close to fixed k=5 accuracy with clearly fewer calls.

### 3.4 Review and commit

Same as 1.4, but ask the reviewer to focus on statistical correctness.

**Done when:** all tests pass and `VALIDATION.md` shows the three results above, or explains honestly why not.

---

## Phase 4: Report and CLI (parallel, about 1 week)

### 4.1 Tests first

```
Phase 4.1. Use the test-writer subagent to write tests for: report rendering (the HTML file is created, is self-contained with no http links to scripts or styles, contains the verdict, every reliable regression case id and every slice tag; report.json round-trips into the models; summary.md contains the verdict and top reasons), and CLI commands using typer's CliRunner with the mock provider: init, run, report, ci exit codes for each verdict level including --strict, runs, import, doctor.
```

### 4.2 Build in parallel

```
Phase 4.2. Run in parallel:
A) report-builder: implement flipcheck/report/ per SPEC 13. Before writing the template, read the frontend-design skill if it is available. Keep it clean, readable and printable, with light and dark mode. No external assets.
B) core-engineer: implement flipcheck/cli.py per SPEC 12, including the init templates that run against the mock target out of the box.
Then run all tests.
```

### 4.3 Look at it yourself

```
Run `uv run flipcheck init` in a temp folder, then `uv run flipcheck run`, then `uv run flipcheck report <id> --open`. Tell me the file path so I can open the report.
```

Open the report and judge it like a user. Can you tell within 10 seconds whether to ship and why? Ask for changes until you can.

### 4.4 Review and commit

Same as 1.4.

**Done when:** a new user can go from `init` to an opened report in under 5 minutes using only the mock target.

---

## Phase 5: CI, docs and dogfooding (about 1 week)

### 5.1 GitHub Action

```
Phase 5.1. Use core-engineer to add .github/workflows/flipcheck-example.yml that installs the package with uv, runs `flipcheck ci` on pull requests, uploads the HTML report as an artifact, and writes the summary to the step summary. It must use the mock target so it needs no secrets, with a commented section showing how to add a real API key as a repository secret. Also add a workflow for the repo's own tests and ruff, plus a step that fails if any tracked text file (.py, .md, .yml, .yaml, .toml, .j2, .html) contains an em dash or en dash. Commit and push, then confirm both workflows pass on GitHub.
```

### 5.2 Dogfood on your own app

Pick one of your existing LLM apps. Build a real suite of 50 to 100 cases from its real inputs with `flipcheck import`, tag them, and mark the cases that must never break as critical. Then compare a real change, such as a prompt edit or a model swap, using free tiers or Ollama.

```
Phase 5.2. Help me create a FlipCheck suite for my app <name>. Here are some real inputs: <paste or path>. Suggest tags and which cases should be critical, then write the suite file and a flipcheck.yaml comparing <baseline> and <candidate>. Keep the run under the free-tier limits in SPEC 4.
```

Write down what it caught. Keep the reports. This is your case study.

### 5.3 README and docs

The README is the first thing people see on your public GitHub. It should be impressive because of real substance, not adjectives.

```
Phase 5.3. Write README.md for the public repo. Follow the writing style and README rules in CLAUDE.md strictly: no em dashes or en dashes, no hype words, and every number either measured by this project or quoted from a paper in SPEC section 16.

Structure:
1. Title, one-line value statement, and badges (CI status, Python version, license, PyPI version once published).
2. A report screenshot or GIF near the top (leave a clearly marked placeholder if it does not exist yet, and tell me).
3. "The problem" in one short paragraph: aggregate accuracy can rise while individual cases break. Cite Yan et al. 2021 for negative flips and MUSCLE 2024 for LLM updates.
4. "Why single runs lie": cite Cacioli 2026 with its exact figures (missed 42% of reliably changed items, falsely flagged 25% of unchanged items) and Ma, Yang and Kästner 2024 (prompt ranking changed by 8.7% after an API update).
5. "What FlipCheck does differently": repeated sampling with per-item reliable change, adaptive sampling, slice reports, a ship or block verdict, CI gate, zero-cost local setup.
6. "Results": a table taken directly from docs/VALIDATION.md (precision, recall, false alarm rate, calls used for single shot vs fixed vs adaptive), plus the real dogfooding result from Phase 5.2 with what it caught. Only real numbers.
7. Quickstart in under 5 minutes with the mock target, then a real example with Ollama.
8. CI setup with the GitHub Action.
9. Configuration reference (short, link to docs for the rest).
10. How it works: one short diagram in Mermaid showing baseline and candidate, repeated runs, scoring, per-item statistics, verdict.
11. Research: the full reference list from SPEC section 16 with links.
12. Roadmap from the backlog, Contributing, License.

After writing, run the dash check on README.md, check every link resolves, and list every number in the README with its source so I can verify. Then commit "phase5.3: write README" and push.
```

### 5.4 Release

```
Phase 5.4. Run the full test suite, ruff, and a final reviewer pass over the whole repo. Fix critical items. Bump the version to 0.1.0, add a CHANGELOG.md, and give me the exact commands to build and publish to TestPyPI first, then PyPI.
```

Publishing on PyPI is free. Publish to TestPyPI first, install from there in a clean folder, then publish for real.

**Done when:** `pip install flipcheck` works in a fresh environment, the example workflow passes on GitHub, and you have one real case study.

---

## After V1 (backlog, not part of this plan)

V2:
- Auto slices: embed inputs, cluster, name clusters with an LLM, suggest tags
- Judge calibration: label 20 to 30 outputs, measure judge agreement, warn when it is low
- Triage loop in the report: accept a change, mark as bug, update expected answer
- Prompt diff view and cost or latency tradeoff in the verdict

V3:
- Agent trace diff: compare multi-step runs step by step and show the first divergence
- Drift watch: scheduled probe set against a live endpoint, alert on behavior change
- Regression risk score per item, using cross-version signals from the 2026 literature
- Hosted dashboard

---

## Timeline at a glance

| Week | Work |
|---|---|
| 1 | Phase 0 and Phase 1 |
| 2 | Phase 2 |
| 3 to 4 | Phase 3, including validation |
| 5 | Phase 4 |
| 6 | Phase 5 |

This assumes part-time work. If a phase slips, cut scope inside the phase (for example CSV and YAML suite loaders, or the http target) rather than skipping review or validation.

## Risks and how to handle them

| Risk | What to do |
|---|---|
| Free tier limits are hit mid-run | Rate limiter plus cache means you can rerun and only missing samples are called. Lower the budget or the suite size |
| Statistics give too few reliable calls on small K | Expected. Raise k_max for critical slices, and show inconclusive items honestly in the report |
| LLM judge is unreliable with small free models | Prefer deterministic scorers. Use the judge only for open answers, and keep judge temperature at 0 |
| Agents drift from the spec | Contracts are orchestrator-owned, reviewer checks every step against the spec, tests come from test-writer |
| Scope creep | Anything not in SPEC goes to the backlog section above |
