"""Configuration loading and version hashing for FlipCheck.

Covers SPEC sections 3 and 4.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from flipcheck.types import ProjectConfig, TargetConfig


def _sort_keys_recursive(obj: Any) -> Any:
    """Return a copy of obj with all dict keys sorted recursively."""
    if isinstance(obj, dict):
        return {k: _sort_keys_recursive(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_sort_keys_recursive(item) for item in obj]
    return obj


def load_config(path: str) -> ProjectConfig:
    """Read and parse a YAML config file into a ProjectConfig.

    Args:
        path: Absolute or relative path to the YAML config file.

    Returns:
        A validated ProjectConfig instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML is invalid or required fields are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = config_path.read_text(encoding="utf-8")

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")

    try:
        return ProjectConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Config file has missing or invalid fields: {exc}") from exc


def compute_version_id(config: TargetConfig, prompt_content: str | None) -> str:
    """Compute a deterministic 12-character hex version ID.

    The ID is derived from the sha256 of the canonical JSON of the config
    dict (keys sorted recursively) concatenated with the prompt content.
    None and empty string produce different IDs.

    Args:
        config: The target configuration to hash.
        prompt_content: The raw text of the prompt file, or None if absent.

    Returns:
        A 12-character lowercase hex string.
    """
    config_dict = _sort_keys_recursive(config.model_dump())
    payload: dict[str, Any] = {
        "config": config_dict,
        "prompt_content": prompt_content,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:12]
