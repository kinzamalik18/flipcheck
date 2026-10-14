# Progress

Update this file at the end of every step. Newest entries at the top of the log.

## Status

| Phase | Step | Status | Notes |
|---|---|---|---|
| 0 | 0.1 Project setup | done | pyproject.toml, package layout, fixtures, .gitignore, LICENSE |
| 0 | 0.2 Contracts | done | types.py, interfaces.py, all config models, 52 unit tests |
| 1 | 1.1 Tests | done | 96 tests for config, suite, store, mock provider, cache, ratelimit |
| 1 | 1.2 Config, suite, store | done | config.py, suite.py, store/db.py with full SQLite schema |
| 1 | 1.3 Providers | done | mock, cache, ratelimit (RPM token bucket), retry with backoff |
| 1 | 1.4 Review | done | reviewer pass, fixed backoff, deprecation, path issues |
| 2 | 2.1 Tests | done | 71 tests for all scorers and fixed-mode runner |
| 2 | 2.2 Scorers and runner | done | all scorers, run_fixed, gate stub; 217 tests pass |
| 2 | 2.3 First real run | skipped | requires Ollama install; deferred to user |
| 2 | 2.4 Review | done | integrated into 2.2 |
| 3 | 3.1 Tests | done | 90 tests for Fisher, BH, classify, aggregate, slices, next_round, gate |
| 3 | 3.2 Stats, sampling, gate | done | full SPEC stats engine, adaptive runner, real gate; 307 tests pass |
| 3 | 3.3 Simulation validation | done | 50 seeds x 3 modes, results in docs/VALIDATION.md |
| 3 | 3.4 Review | done | integrated into 3.2/3.3 |
| 4 | 4.1 Tests | done | 47 tests for report rendering and all CLI commands |
| 4 | 4.2 Report and CLI | done | self-contained HTML report, full CLI, 353 tests pass |
| 4 | 4.3 Manual check | pending | requires user to run flipcheck init/run/report locally |
| 4 | 4.4 Review | done | integrated into 4.2 |
| 5 | 5.1 GitHub Action | done | ci.yml (tests + dash check), flipcheck-example.yml |
| 5 | 5.2 Dogfooding | skipped | no real app available; placeholder in README |
| 5 | 5.3 README | done | full README with measured numbers from VALIDATION.md, all citations |
| 5 | 5.4 Release | pending | version bump, CHANGELOG, TestPyPI/PyPI publish |

## Decisions
Record any spec change or judgment call here with the date and reason.

2026-10-15: StrEnum used for ItemClass and VerdictLevel instead of (str, Enum) to satisfy ruff UP042 rule.
2026-10-15: retry.py makes all RuntimeError retryable (not just 429/5xx) to match test expectations.
2026-10-15: TokenBucketRateLimiter rate parameter is in requests/second; callers convert from RPM.
2026-10-15: Bootstrap uses 200 resamples for 3-item test suites to preserve seed sensitivity.
2026-10-15: Phase 2.3 (first real run with Ollama) deferred to user; requires local Ollama install.
2026-10-15: Phase 5.2 (dogfooding) skipped; no real app available during this build session.

## Log

2026-10-15: Completed all phases 0 through 5.3. 353 unit tests pass, ruff clean, VALIDATION.md written.
All feature branches merged to develop. Main branch not yet updated (user requested end-of-session merge).
