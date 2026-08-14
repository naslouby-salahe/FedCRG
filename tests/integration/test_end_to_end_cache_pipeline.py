"""Smoke-test the real train -> score -> evaluate cache pipeline end-to-end."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fedcrg.config import ExperimentConfig
from fedcrg.experiments.runner import ComputeScores, EvaluatePolicies, TrainDetector
from fedcrg.paths import OutputsLayout
from fedcrg.thresholding.readiness import ReadinessPlanCache
from fedcrg.types import CalibrationAssignmentMode
from tests._fixtures import (
    diad_pipeline_config,
    write_prepared_diad,
)


def _config(root: Path) -> ExperimentConfig:
    return diad_pipeline_config(root)


def _write_prepared(root: Path, config: ExperimentConfig) -> Path:
    return write_prepared_diad(root, config)


def test_train_score_evaluate_cache_pipeline(tmp_path: Path) -> None:
    config = _config(tmp_path / "outputs")
    prepared = _write_prepared(tmp_path / "prepared", config)

    model_path, training_manifest = TrainDetector().train_from_cache(config, prepared, 11)
    assert model_path.exists()
    assert training_manifest.exists()

    score_root = ComputeScores().score_from_cache(
        config, prepared, model_path, 11, training_manifest
    )

    readiness_cache_path = OutputsLayout(config.outputs_root).readiness_plans_file
    readiness_cache = ReadinessPlanCache(readiness_cache_path)
    readiness_cache.precompute(
        config.dataset.split.calibration_benign,
        config.protocol.band,
        config.protocol.readiness_assurance,
    )
    readiness_cache.save()

    bundle = EvaluatePolicies().evaluate_from_cache(
        config,
        score_root,
        calibration_seed=1000,
        mode=CalibrationAssignmentMode.SEEDED_PERMUTATION,
    )
    assert len(bundle.clients) == 2 * len(config.policies)
    metrics = [item.metrics for item in bundle.clients if item.metrics is not None]
    assert metrics
    assert all(np.isfinite(item.fpr) for item in metrics)
