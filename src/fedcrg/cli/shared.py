"""Shared CLI boundary helpers."""

from __future__ import annotations

from pathlib import Path

from fedcrg.config.models import ExperimentConfig
from fedcrg.config.resolver import ExperimentConfigResolver
from fedcrg.config.validation import validate_experiment_config


def load_config(path: Path) -> ExperimentConfig:
    config = ExperimentConfigResolver().resolve(path)
    validate_experiment_config(config)
    return config
