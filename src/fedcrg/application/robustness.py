"""Robustness application entry points."""

from __future__ import annotations

from pathlib import Path

from fedcrg.application.train import TrainDetector
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import DetectorId


class RunRobustness:
    def train_second_detector(
        self,
        config: ExperimentConfig,
        prepared_root: Path,
        model_seed: int,
    ) -> tuple[Path, Path]:
        if config.detector.id is not DetectorId.DEEP_SVDD:
            raise ValueError("Second-detector robustness requires the Deep-SVDD config")
        return TrainDetector().train_from_cache(config, prepared_root, model_seed)
