"""Tests for flipcheck/config.py: YAML loading and version hashing.

Covers SPEC sections 3 and 4.
"""

from __future__ import annotations

import textwrap

import pytest

# Import the module under test. Skip all tests in this file if the symbols
# are not yet present (stub modules that do not implement the function yet).
import flipcheck.config as _config_mod
from flipcheck.types import ProjectConfig, TargetConfig

if not hasattr(_config_mod, "load_config"):
    pytest.skip("flipcheck.config.load_config not implemented yet", allow_module_level=True)

load_config = _config_mod.load_config
compute_version_id = _config_mod.compute_version_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_YAML = textwrap.dedent(
    """\
    project: my-llm-app
    suite: suites/main.jsonl
    targets:
      baseline:
        type: mock
    """
)

FULL_YAML = textwrap.dedent(
    """\
    project: my-llm-app
    suite: suites/main.jsonl
    default_scorers:
      - type: exact
    targets:
      baseline:
        type: llm
        model: ollama/llama3.2:3b
        prompt: prompts/v1.j2
        params:
          temperature: 0.7
          max_tokens: 300
      candidate:
        type: llm
        model: groq/openai/gpt-oss-120b
        prompt: prompts/v2.j2
        params:
          temperature: 0.7
          max_tokens: 300
    judge:
      model: gemini/gemini-2.5-flash
      params:
        temperature: 0
    sampling:
      mode: adaptive
      k: 5
      k_min: 2
      k_max: 8
      step: 2
      budget_calls: 1500
    gates:
      alpha: 0.05
      max_reliable_nfr: 0.03
      block_on_critical_regression: true
      max_flaky_rate: 0.15
      max_inconclusive_rate: 0.10
      max_slice_drop: 0.10
    providers:
      rate_limits:
        groq: 30
        gemini: 10
        ollama: 0
      concurrency: 4
    cache_path: .flipcheck/flipcheck.sqlite
    """
)


# ---------------------------------------------------------------------------
# Helper: write a YAML string to a temp file and return its path
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path, content: str, filename: str = "flipcheck.yaml") -> str:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# load_config: minimal YAML
# ---------------------------------------------------------------------------


def test_load_minimal_yaml_returns_project_config(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(path)
    assert isinstance(cfg, ProjectConfig)


def test_load_minimal_yaml_project_name(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(path)
    assert cfg.project == "my-llm-app"


def test_load_minimal_yaml_suite_path(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(path)
    assert cfg.suite == "suites/main.jsonl"


def test_load_minimal_yaml_has_baseline_target(tmp_path):
    path = _write_yaml(tmp_path, MINIMAL_YAML)
    cfg = load_config(path)
    assert "baseline" in cfg.targets
    assert cfg.targets["baseline"].type == "mock"


# ---------------------------------------------------------------------------
# load_config: full YAML
# ---------------------------------------------------------------------------


def test_load_full_yaml_returns_project_config(tmp_path):
    path = _write_yaml(tmp_path, FULL_YAML)
    cfg = load_config(path)
    assert isinstance(cfg, ProjectConfig)


def test_load_full_yaml_both_targets(tmp_path):
    path = _write_yaml(tmp_path, FULL_YAML)
    cfg = load_config(path)
    assert "baseline" in cfg.targets
    assert "candidate" in cfg.targets


def test_load_full_yaml_sampling_mode(tmp_path):
    path = _write_yaml(tmp_path, FULL_YAML)
    cfg = load_config(path)
    assert cfg.sampling.mode == "adaptive"


def test_load_full_yaml_sampling_budget(tmp_path):
    path = _write_yaml(tmp_path, FULL_YAML)
    cfg = load_config(path)
    assert cfg.sampling.budget_calls == 1500


def test_load_full_yaml_gate_alpha(tmp_path):
    path = _write_yaml(tmp_path, FULL_YAML)
    cfg = load_config(path)
    assert cfg.gates.alpha == pytest.approx(0.05)


def test_load_full_yaml_judge_model(tmp_path):
    path = _write_yaml(tmp_path, FULL_YAML)
    cfg = load_config(path)
    assert cfg.judge.model == "gemini/gemini-2.5-flash"


def test_load_full_yaml_providers_concurrency(tmp_path):
    path = _write_yaml(tmp_path, FULL_YAML)
    cfg = load_config(path)
    assert cfg.providers.concurrency == 4


def test_load_full_yaml_default_scorers(tmp_path):
    path = _write_yaml(tmp_path, FULL_YAML)
    cfg = load_config(path)
    assert len(cfg.default_scorers) == 1
    assert cfg.default_scorers[0].type == "exact"


def test_load_full_yaml_rate_limits(tmp_path):
    path = _write_yaml(tmp_path, FULL_YAML)
    cfg = load_config(path)
    assert cfg.providers.rate_limits.groq == 30
    assert cfg.providers.rate_limits.ollama == 0


# ---------------------------------------------------------------------------
# compute_version_id: basic contract
# ---------------------------------------------------------------------------


def test_version_id_is_12_char_hex():
    cfg = TargetConfig(type="mock", mock={"pass_prob": {"*": 0.8}, "seed": 1})
    vid = compute_version_id(cfg, prompt_content=None)
    assert len(vid) == 12
    assert all(c in "0123456789abcdef" for c in vid)


def test_version_id_same_config_no_prompt_is_deterministic():
    cfg = TargetConfig(type="mock", mock={"pass_prob": {"*": 0.8}, "seed": 1})
    assert compute_version_id(cfg, prompt_content=None) == compute_version_id(
        cfg, prompt_content=None
    )


def test_version_id_same_config_same_prompt_is_deterministic():
    cfg = TargetConfig(
        type="llm",
        model="ollama/llama3.2:3b",
        prompt="prompts/v1.j2",
    )
    prompt = "You are a helpful assistant. {{query}}"
    assert compute_version_id(cfg, prompt_content=prompt) == compute_version_id(
        cfg, prompt_content=prompt
    )


def test_version_id_different_configs_give_different_ids():
    cfg_a = TargetConfig(type="mock", mock={"pass_prob": {"*": 0.8}, "seed": 1})
    cfg_b = TargetConfig(type="mock", mock={"pass_prob": {"*": 0.5}, "seed": 1})
    assert compute_version_id(cfg_a, prompt_content=None) != compute_version_id(
        cfg_b, prompt_content=None
    )


def test_version_id_different_prompt_gives_different_id():
    cfg = TargetConfig(
        type="llm",
        model="ollama/llama3.2:3b",
        prompt="prompts/v1.j2",
    )
    id_a = compute_version_id(cfg, prompt_content="Prompt A {{query}}")
    id_b = compute_version_id(cfg, prompt_content="Prompt B {{query}}")
    assert id_a != id_b


def test_version_id_none_vs_empty_prompt_differ():
    # A prompt file that is absent (None) and an empty string prompt are distinct.
    cfg = TargetConfig(type="llm", model="ollama/llama3.2:3b")
    assert compute_version_id(cfg, prompt_content=None) != compute_version_id(
        cfg, prompt_content=""
    )


# ---------------------------------------------------------------------------
# load_config: error handling
# ---------------------------------------------------------------------------


def test_load_config_invalid_yaml_raises_clear_error(tmp_path):
    path = _write_yaml(tmp_path, "targets: [\nunclosed bracket")
    with pytest.raises(Exception, match=r"(?i)(invalid|yaml|parse|error)"):
        load_config(path)


def test_load_config_missing_targets_raises_clear_error(tmp_path):
    yaml_no_targets = textwrap.dedent(
        """\
        project: my-llm-app
        suite: suites/main.jsonl
        """
    )
    path = _write_yaml(tmp_path, yaml_no_targets)
    with pytest.raises((ValueError, KeyError, TypeError, SystemExit)):
        load_config(path)


def test_load_config_missing_project_raises_error(tmp_path):
    yaml_no_project = textwrap.dedent(
        """\
        suite: suites/main.jsonl
        targets:
          baseline:
            type: mock
        """
    )
    path = _write_yaml(tmp_path, yaml_no_project)
    with pytest.raises((ValueError, KeyError, TypeError, SystemExit)):
        load_config(path)


def test_load_config_nonexistent_file_raises_error(tmp_path):
    with pytest.raises((FileNotFoundError, OSError, ValueError, SystemExit)):
        load_config(str(tmp_path / "does_not_exist.yaml"))
