"""Shared CLI boundary helpers."""

from __future__ import annotations

from pathlib import Path

from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.config.resolve import ExperimentConfigResolver
from fedcrg.config.validate import validate_experiment_config


def load_config(path: Path) -> ExperimentConfig:
    config = ExperimentConfigResolver().resolve(path)
    validate_experiment_config(config)
    return config
