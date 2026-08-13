"""YAML loading utilities with explicit untyped-data boundaries."""

from __future__ import annotations

from pathlib import Path

import yaml

from fedcrg.domain.errors import ConfigurationError


def load_yaml(path: Path) -> dict[str, object]:
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ConfigurationError(f"Configuration root must be a string-keyed mapping: {path}")
    return {str(key): value for key, value in raw.items()}
