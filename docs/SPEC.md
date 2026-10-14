# FlipCheck V1 Technical Spec

FlipCheck is a working name. Rename it freely; only update `pyproject.toml`, the package folder and this file.

## 1. Purpose and scope

FlipCheck compares two versions of an LLM-powered app (the **baseline** and the **candidate**) on the same test suite and decides whether the change is safe to ship. A version is any combination of model, prompt and parameters.

What makes it better than a plain eval runner:

1. **Per-item reliable change.** Every test runs several times per version. Each item is classified as a real regression, a real improvement, unchanged, or flaky. Flaky items never fail a build on their own.
2. **Adaptive sampling.** Extra samples are spent only on items whose result is still uncertain, which keeps API cost low.
3. **Slice reports.** Results are broken down by tags such as `urdu` or `emergency`, and critical cases are gated more strictly.
4. **A clear verdict.** Every run ends in SHIP, SHIP_WITH_CAVEATS or BLOCK, with the evidence behind it.

### V1 runs on zero budget
- Local models through Ollama for development
- Free API tiers (Gemini, Groq) through LiteLLM
- Bring your own key: users supply their own API keys
- SQLite storage, a static HTML report, GitHub Actions for CI

### Not in V1
Hosted multi-user dashboard, accounts, auto slice discovery (V2), judge calibration (V2), agent trace diff (V3), drift watch (V3), regression risk prediction (V3).

## 2. Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Packaging | uv, `pyproject.toml` |
| CLI | Typer + rich |
| Models and config | Pydantic v2 |
| LLM access | LiteLLM (async `acompletion`) |
| HTTP targets | httpx (async) |
| Storage | SQLite through the stdlib `sqlite3` module, thin repository layer |
| Statistics | numpy, scipy |
| Semantic similarity | sentence-transformers (optional extra `[semantic]`) |
| Schema checks | jsonschema |
| Reports | Jinja2, one self-contained HTML file |
| Tests and lint | pytest, pytest-asyncio, ruff |

No other runtime dependency without approval from the orchestrator.

## 3. Core concepts (the contract in `flipcheck/types.py`)

All models are Pydantic v2. Names below are binding for every module.

```python
class TargetConfig(BaseModel):
    type: Literal["llm", "http", "python", "mock"]
    model: str | None = None            # LiteLLM model string, for type=llm
    prompt: str | None = None           # path to a prompt template file (Jinja2), for type=llm
    system: str | None = None           # optional path to a system prompt file
    params: dict[str, Any] = {}         # temperature, max_tokens, etc.
    url: str | None = None              # for type=http
    callable: str | None = None         # "package.module:function", for type=python
    mock: dict[str, Any] | None = None  # for type=mock, see section 6

class Version(BaseModel):
    name: str                           # "baseline" or "candidate" or a user label
    config: TargetConfig
    version_id: str                     # sha256 of canonical JSON of config + prompt file contents, first 12 chars

class ScorerConfig(BaseModel):
    type: Literal["exact", "contains", "regex", "json_schema", "json_field",
                  "semantic", "llm_judge", "python"]
    args: dict[str, Any] = {}

class TestCase(BaseModel):
    id: str
    input: dict[str, Any]               # variables for the prompt template, or the request body
    expected: Any | None = None
    scorers: list[ScorerConfig] = []    # falls back to the suite default scorers when empty
    tags: list[str] = []
    critical: bool = False

class Sample(BaseModel):
    case_id: str
    version_id: str
    sample_idx: int
    output: str | None
    error: str | None = None
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cached: bool = False

class Score(BaseModel):
    scorer: str
    value: float                        # 0.0 to 1.0
    passed: bool
    rationale: str | None = None

class ItemCounts(BaseModel):
    case_id: str
    base_pass: int
    base_n: int
    cand_pass: int
    cand_n: int
    base_first_pass: bool | None        # result of sample_idx 0, for single-shot NFR
    cand_first_pass: bool | None

class ItemClass(str, Enum):
    RELIABLE_REGRESSION = "reliable_regression"
    RELIABLE_IMPROVEMENT = "reliable_improvement"
    UNCHANGED_PASS = "unchanged_pass"
    UNCHANGED_FAIL = "unchanged_fail"
    FLAKY = "flaky"
    INCONCLUSIVE = "inconclusive"

class ItemResult(BaseModel):
    counts: ItemCounts
    classification: ItemClass
    p_regression: float | None          # BH-adjusted
    p_improvement: float | None         # BH-adjusted

class SliceSummary(BaseModel):
    tag: str
    n_items: int
    base_rate: float
    cand_rate: float
    delta: float
    regressions: int
    improvements: int
    low_confidence: bool                # n_items < 10

class RunSummary(BaseModel):
    n_items: int
    base_rate: float
    cand_rate: float
    delta: float
    delta_ci: tuple[float, float]       # paired bootstrap, 95 percent
    mcnemar_p: float | None
    nfr_single_shot: float              # negative flips on sample 0 / n_items
    nfr_single_shot_conditional: float  # same, over items the baseline got right on sample 0
    reliable_nfr: float                 # reliable regressions / n_items
    churn_rate: float                   # (reliable regressions + reliable improvements) / n_items
    flaky_rate: float
    inconclusive_rate: float
    slices: list[SliceSummary]
    calls_used: int
    est_cost_usd: float | None

class VerdictLevel(str, Enum):
    SHIP = "ship"
    SHIP_WITH_CAVEATS = "ship_with_caveats"
    BLOCK = "block"

class Reason(BaseModel):
    level: VerdictLevel
    message: str                        # plain English, one sentence
    case_ids: list[str] = []

class Verdict(BaseModel):
    level: VerdictLevel
    reasons: list[Reason]
```

`flipcheck/interfaces.py` holds the Protocols:

```python
class Provider(Protocol):
    async def generate(self, version: Version, case: TestCase, sample_idx: int) -> Sample: ...

class Scorer(Protocol):
    name: str
    async def score(self, case: TestCase, output: str) -> Score: ...

class Store(Protocol):  # see section 9 for methods
    ...
```

## 4. Config file

`flipcheck.yaml` in the user's project:

```yaml
project: my-llm-app
suite: suites/main.jsonl
default_scorers:
  - type: exact
targets:
  baseline:
    type: llm
    model: ollama/llama3.2:3b          # check exact model strings in the LiteLLM docs
    prompt: prompts/v1.j2
    params: {temperature: 0.7, max_tokens: 300}
  candidate:
    type: llm
    model: groq/openai/gpt-oss-120b
    prompt: prompts/v2.j2
    params: {temperature: 0.7, max_tokens: 300}
judge:
  model: gemini/gemini-2.5-flash
  params: {temperature: 0}
sampling:
  mode: adaptive                       # fixed or adaptive
  k: 5                                 # used when mode is fixed
  k_min: 2
  k_max: 8
  step: 2
  budget_calls: 1500                   # hard cap on model calls, judge included
gates:
  alpha: 0.05
  max_reliable_nfr: 0.03
  block_on_critical_regression: true
  max_flaky_rate: 0.15
  max_inconclusive_rate: 0.10
  max_slice_drop: 0.10
providers:
  rate_limits:                         # requests per minute per LiteLLM provider prefix
    groq: 30
    gemini: 10
    ollama: 0                          # 0 means no limit
  concurrency: 4
cache_path: .flipcheck/flipcheck.sqlite
```

API keys come only from environment variables or a `.env` file. They are never written to config, the database or reports.

## 5. Test suite format

JSONL, one case per line. CSV and YAML loaders map to the same model.

```json
{"id": "t001", "input": {"query": "Mujhe seene mein dard hai"}, "expected": "urgent",
 "scorers": [{"type": "json_field", "args": {"field": "label"}}],
 "tags": ["urdu", "emergency"], "critical": true}
```

Rules: ids are unique, `input` keys must match the template variables, unknown scorer types fail at load time with a clear message.

## 6. Providers and targets

- `llm`: renders the Jinja2 prompt with `case.input`, calls `litellm.acompletion`. Records latency and token usage.
- `http`: POSTs `case.input` as JSON to `url`, reads `output` from the response body (configurable JSON path).
- `python`: imports the callable and awaits it if it is async.
- `mock`: deterministic and seedable, used by all tests. Config `{"pass_prob": {"t001": 0.9, "*": 0.8}, "seed": 1}`. It returns the string `"PASS"` or `"FAIL"` drawn with that probability from `random.Random(hash(seed, version_id, case_id, sample_idx))`. Scored with the `exact` scorer against `expected: "PASS"`.

Shared behavior:
- Output cache keyed by `sha256(version_id, case_id, sample_idx)`. A cache hit sets `cached=True` and costs no call.
- Token-bucket rate limiter per provider prefix, plus a global concurrency semaphore.
- Retries on 429 and 5xx: 3 attempts, exponential backoff with jitter. After that the sample gets `error` set and counts as a failed sample, and the error is shown in the report.

## 7. Scorers

All scorers return a `Score`. A sample passes only if every scorer for the case passes.

| Type | Args | Behavior |
|---|---|---|
| exact | `field` (optional), `normalize` (default true) | Case-insensitive, trimmed comparison with `expected` |
| contains | `values`, `mode` (all or any) | Substring checks |
| regex | `pattern`, `must_match` (default true) | Regex search |
| json_schema | `schema` (inline or path) | Output must parse as JSON and validate |
| json_field | `field`, `normalize` | Parse JSON, compare one field to `expected` |
| semantic | `threshold` (default 0.8), `model` | Cosine similarity with `expected` using sentence-transformers. Model loads lazily, once per process |
| llm_judge | `rubric`, `pass_threshold` | Calls the judge model with the rubric, the input, the expected answer if any, and the output. The judge must return JSON `{"score": 0-1, "pass": bool, "reason": str}`. Invalid JSON gets one retry, then a failed score with the parse error as rationale. Judge results are cached by `sha256(judge config, rubric, case_id, output)` |
| python | `callable` | User function `(case, output) -> Score or bool` |

Judge calls count toward `budget_calls`.

## 8. Runner and sampling

### 8.1 Fixed mode
Run `k` samples per case for each version, score them, build `ItemCounts`.

### 8.2 Adaptive mode
1. Run `k_min` samples per case for both versions.
2. Classify every item (section 9.1). An item is settled when it is `reliable_regression`, `reliable_improvement`, or when both versions are unanimous and agree (all pass or all fail on both sides).
3. For unsettled items, add `step` samples to both versions, then reclassify. Stop for an item at `k_max`.
4. When the remaining budget cannot cover every unsettled item, order them as: critical items first, then items with the largest current difference in pass rate, then by id.
5. When the budget runs out, any still-unsettled item with fewer than `k_max` samples is `inconclusive`.

The adaptive policy is a pure function in `flipcheck/sampling/adaptive.py`:

```python
def next_round(counts: list[ItemCounts], cfg: SamplingConfig, budget_left: int,
               critical: set[str]) -> dict[str, int]:
    """Return case_id -> extra samples per version for the next round. Empty dict means stop."""
```

Repeated looks at the data raise the false positive rate. Phase 3 measures this by simulation and tunes `alpha` handling if needed (for example a stricter per-look alpha).

## 9. Statistics (`flipcheck/stats/`)

All functions are pure and take plain data, so they are easy to test.

### 9.1 Per-item classification
For each item: baseline `a` passes out of `n1`, candidate `b` out of `n2`.
- `p_regression`: one-sided Fisher exact test that the candidate pass rate is lower.
- `p_improvement`: one-sided Fisher exact test that it is higher.
- Apply Benjamini-Hochberg separately to the regression and improvement p-values across all items.
- Classification, in order:
  1. adjusted `p_regression < alpha` gives RELIABLE_REGRESSION
  2. adjusted `p_improvement < alpha` gives RELIABLE_IMPROVEMENT
  3. budget ran out before `k_max` and not settled gives INCONCLUSIVE
  4. either pass rate between 0.2 and 0.8 inclusive gives FLAKY
  5. both pass rates at least 0.5 gives UNCHANGED_PASS
  6. otherwise UNCHANGED_FAIL

Known limit: with 5 samples per side the smallest Fisher p-value is about 0.004, so with many items BH can make reliable calls hard. This is expected and is why adaptive sampling goes up to `k_max`.

### 9.2 Aggregates
- Pass rate per item = passes / n. Version rate = mean over items.
- `delta` = cand_rate minus base_rate. `delta_ci` = paired bootstrap over items, 2000 resamples, fixed seed.
- `mcnemar_p`: exact McNemar (binomial) on majority-vote pass per item. Ties at exactly 50 percent count as fail.
- `nfr_single_shot` and `nfr_single_shot_conditional` use sample 0 only, so numbers line up with the literature (Yan et al. 2021 define NFR over all items; Xie et al. 2021 and later work also use the subset the old model got right).
- `reliable_nfr`, `churn_rate`, `flaky_rate`, `inconclusive_rate` as defined in section 3.

### 9.3 Slices
For each tag: items with that tag, base and cand rates, delta, counts of reliable regressions and improvements, `low_confidence` when fewer than 10 items. A synthetic slice `critical` covers all critical cases.

## 10. Gate and verdict (`flipcheck/gate.py`)

```python
def decide(summary: RunSummary, items: list[ItemResult], critical: set[str],
           gates: GateConfig) -> Verdict
```

BLOCK when any of these hold:
- a critical case is a reliable regression and `block_on_critical_regression` is true
- `reliable_nfr > max_reliable_nfr`
- the upper bound of `delta_ci` is below 0 (overall quality dropped)

SHIP_WITH_CAVEATS when not blocked and any of these hold:
- at least one non-critical reliable regression
- `flaky_rate > max_flaky_rate`
- `inconclusive_rate > max_inconclusive_rate`
- a slice that is not low confidence has `delta < -max_slice_drop`

SHIP otherwise. Every triggered rule adds a `Reason` with a one-sentence message and the case ids involved. Reasons are sorted BLOCK first, then by number of cases.

## 11. Storage (`flipcheck/store/`)

SQLite file at `cache_path`. Tables:

- `versions(version_id PK, name, config_json, created_at)`
- `runs(run_id PK, project, suite_hash, baseline_id, candidate_id, config_json, status, started_at, finished_at, calls_used, est_cost_usd)`
- `cases(run_id, case_id, input_json, expected_json, tags_json, critical, PK(run_id, case_id))`
- `samples(sample_pk PK, run_id, version_id, case_id, sample_idx, output, error, latency_ms, tokens_in, tokens_out, cached)`
- `scores(sample_pk, scorer, value, passed, rationale)`
- `item_results(run_id, case_id, classification, base_pass, base_n, cand_pass, cand_n, p_regression, p_improvement)`
- `run_summaries(run_id PK, summary_json, verdict_json)`
- `output_cache(key PK, sample_json, created_at)` and `judge_cache(key PK, score_json, created_at)`

Schema version stored in `PRAGMA user_version`; migrations are plain SQL files applied in order. The repository class exposes typed methods (`save_sample`, `get_samples(run_id, case_id, version_id)`, `save_item_results`, `list_runs`, and so on) and is the only code that writes SQL.

## 12. CLI (`flipcheck/cli.py`)

| Command | What it does |
|---|---|
| `flipcheck init` | Writes a sample `flipcheck.yaml`, `suites/example.jsonl` and two prompt files that work with the mock target |
| `flipcheck run [--config PATH] [--fixed-k N] [--budget N]` | Runs a comparison, prints a rich summary and the run id |
| `flipcheck report RUN_ID [--out PATH] [--open]` | Writes `report.html`, `report.json`, `summary.md` |
| `flipcheck ci [--strict]` | Runs, writes the reports, appends `summary.md` to `$GITHUB_STEP_SUMMARY` when set. Exit code 1 on BLOCK, also on SHIP_WITH_CAVEATS with `--strict`, otherwise 0 |
| `flipcheck runs` | Lists past runs with verdicts |
| `flipcheck import FILE --input-field X [--expected-field Y] [--tag T]` | Turns a JSONL log of real traffic into a suite file |
| `flipcheck doctor` | Checks config, prompt files, env keys and Ollama reachability, without spending calls |

## 13. Report

One self-contained HTML file. Inline CSS and JS, no CDN, so it works offline and on GitHub Pages.

Sections, in order:
1. Verdict banner (SHIP green, CAVEATS amber, BLOCK red) and the top reasons in plain English
2. Key numbers: base vs cand rate with CI, reliable NFR, churn, flaky rate, calls and estimated cost
3. Slice table, sorted by delta, low-confidence slices marked
4. Reliable regressions: per case, the input, the expected answer, each sample's output for both versions side by side, pass marks and scorer rationales
5. Flaky and inconclusive items, collapsed by default
6. Reliable improvements, collapsed
7. Run metadata: both version configs with the prompt diff, sampling settings, timestamps

`report.json` is the full `RunSummary`, `Verdict` and item list. `summary.md` is a short markdown version for CI.

## 14. Package layout

```
flipcheck/
  __init__.py
  types.py              # contract, orchestrator-owned
  interfaces.py         # contract, orchestrator-owned
  config.py             # yaml loading, version hashing
  suite.py              # suite loaders, import from logs
  providers/
    __init__.py         # get_provider(version)
    llm.py  http.py  python_target.py  mock.py
    cache.py  ratelimit.py  retry.py
  scorers/
    __init__.py         # build_scorers(case, defaults)
    basic.py            # exact, contains, regex
    structured.py       # json_schema, json_field
    semantic.py  judge.py  custom.py
  sampling/
    adaptive.py
  stats/
    item.py  aggregate.py  slices.py
  gate.py
  runner.py
  store/
    db.py  migrations/
  report/
    render.py  templates/report.html.j2
  cli.py
tests/
  unit/  integration/  simulation/
scripts/
  simulate.py
docs/
  SPEC.md  PLAN.md  PROGRESS.md  VALIDATION.md
```

## 15. Testing rules

- Unit tests never touch the network. Use the mock provider and a fake judge.
- Statistics are tested against hand-computed cases and against scipy where possible.
- `tests/simulation/` checks behavior on synthetic suites with known ground truth (section 8.2 and Phase 3 of the plan). These may be slow and are marked `@pytest.mark.slow`.
- Optional live smoke tests against Ollama are marked `@pytest.mark.live` and skipped by default.

## 16. Research references

Use these in the README and docs. Quote only the figures listed here, and check each link before publishing.

| Paper | Venue | What it established | Figure you may quote |
|---|---|---|---|
| Yan et al., *Positive-Congruent Training: Towards Regression-Free Model Updates* ([arXiv 2011.09161](https://arxiv.org/abs/2011.09161)) | CVPR 2021 | Named "negative flips" and the Negative Flip Rate (NFR); showed that even retraining the same architecture causes regressions | None needed, cite the concept |
| Xie et al., *Regression Bugs Are In Your Model! Measuring, Reducing and Analyzing Regressions in NLP Model Updates* | ACL 2021 | Brought negative flips to NLP model updates | None needed, cite the concept |
| Echterhoff et al., *MUSCLE: A Model Update Strategy for Compatible LLM Evolution* ([arXiv 2407.09435](https://arxiv.org/abs/2407.09435)) | 2024 | Showed negative flips happen when an LLM base model is updated, even with identical downstream training | Negative flips reduced by up to 40% from Llama 1 to Llama 2 with their method |
| Ma, Yang, Kästner, *(Why) Is My Prompt Getting Worse? Rethinking Regression Testing for Evolving LLM APIs* ([arXiv 2311.11123](https://arxiv.org/abs/2311.11123)) | CAIN 2024 | LLM APIs change silently; regression tests must track model and prompt versions together and handle non-determinism | After an API update, a different prompt became best by 8.7% accuracy |
| Cacioli, *Beyond the Mean: Within-Model Reliable Change Detection for LLM Evaluation* ([arXiv 2604.27405](https://arxiv.org/abs/2604.27405)) | 2026 | Adapted the Reliable Change Index to item-level LLM version comparison using repeated sampling | Greedy single-shot evaluation missed 42% of reliably changed items and falsely flagged 25% of unchanged items |
| *CASPER: Change-Aware Slice Prioritization for Efficient Regression Testing of LLM-based Systems* ([arXiv 2608.00378](https://arxiv.org/abs/2608.00378)) | 2026 | Regression analysis works better at the level of test suite slices | None needed |
| *No Universal Signal Predicts Sample-Level LLM Regression under Version Updates* ([arXiv 2608.13607](https://arxiv.org/abs/2608.13607)) | 2026 | Which signals predict negative flips depends on the task | None needed |
| *Test Before You Deploy: Governing Updates in the LLM Supply Chain* ([arXiv 2604.27789](https://arxiv.org/abs/2604.27789)) | 2026 | Argues for deployer-side compatibility gates that block updates | None needed |

