"""Behavior contract: model and score caches are reused by identity.

Training and scoring must be skipped when the cache identity already exists:
identical requests return the existing cache root untouched, calibration-seed
changes must not invalidate the model or score caches, and a training-spec
change must select a distinct model cache while leaving the original intact.
"""

from __future__ import annotations

from pathlib import Path

from fedcrg.config import ExperimentConfig
from fedcrg.evidence.store import OutputsLayout
from fedcrg.experiments.runner import ComputeScores, TrainDetector
from tests._fixtures import diad_pipeline_config, write_prepared_diad


def _train_once(
    config: ExperimentConfig, prepared: Path, model_seed: int = 11
) -> tuple[Path, Path]:
    return TrainDetector().train_from_cache(config, prepared, model_seed)


def test_identical_training_request_reuses_model_cache(tmp_path: Path) -> None:
    config = diad_pipeline_config(tmp_path)
    prepared = write_prepared_diad(tmp_path / "prepared", config)
    model_path, _ = _train_once(config, prepared)
    second_path, _ = _train_once(config, prepared)
    assert second_path == model_path
    assert model_path.parent.is_dir()


def test_calibration_seed_change_keeps_model_cache(tmp_path: Path) -> None:
    config = diad_pipeline_config(tmp_path)
    prepared = write_prepared_diad(tmp_path / "prepared", config)
    model_path, _ = _train_once(config, prepared)
    changed = config.model_copy(
        deep=True,
        update={"dataset": config.dataset.model_copy(update={"calibration_seeds": (1000, 1001)})},
    )
    second_path, _ = _train_once(changed, prepared)
    assert second_path == model_path


def test_training_spec_change_selects_distinct_model_cache(tmp_path: Path) -> None:
    config = diad_pipeline_config(tmp_path)
    prepared = write_prepared_diad(tmp_path / "prepared", config)
    model_path, _ = _train_once(config, prepared)
    assert config.training is not None
    changed = config.model_copy(
        deep=True,
        update={"training": config.training.model_copy(update={"local_epochs": 2})},
    )
    assert changed.training_spec_hash != config.training_spec_hash
    second_path, _ = _train_once(changed, prepared)
    assert second_path != model_path
    assert model_path.parent.is_dir() and second_path.parent.is_dir()


def test_identical_scoring_request_reuses_score_cache(tmp_path: Path) -> None:
    config = diad_pipeline_config(tmp_path)
    prepared = write_prepared_diad(tmp_path / "prepared", config)
    model_path, training_manifest = _train_once(config, prepared)
    compute = ComputeScores()
    first = compute.score_from_cache(config, prepared, model_path, 11, training_manifest)
    second = compute.score_from_cache(config, prepared, model_path, 11, training_manifest)
    assert first == second


def test_calibration_seed_change_keeps_score_cache(tmp_path: Path) -> None:
    config = diad_pipeline_config(tmp_path)
    prepared = write_prepared_diad(tmp_path / "prepared", config)
    model_path, training_manifest = _train_once(config, prepared)
    compute = ComputeScores()
    first = compute.score_from_cache(config, prepared, model_path, 11, training_manifest)
    changed = config.model_copy(
        deep=True,
        update={"dataset": config.dataset.model_copy(update={"calibration_seeds": (1000, 1001)})},
    )
    second = compute.score_from_cache(changed, prepared, model_path, 11, training_manifest)
    assert second == first
    assert OutputsLayout(changed.outputs_root).score_root(changed, 11) == OutputsLayout(
        config.outputs_root
    ).score_root(config, 11)


def test_model_change_invalidates_dependent_score_cache(tmp_path: Path) -> None:
    config = diad_pipeline_config(tmp_path)
    prepared = write_prepared_diad(tmp_path / "prepared", config)
    model_path, training_manifest = _train_once(config, prepared)
    compute = ComputeScores()
    first = compute.score_from_cache(config, prepared, model_path, 11, training_manifest)

    assert config.training is not None
    changed = config.model_copy(
        deep=True,
        update={"training": config.training.model_copy(update={"local_epochs": 2})},
    )
    changed_model_path, changed_manifest = _train_once(changed, prepared)
    assert changed_model_path != model_path
    second = compute.score_from_cache(changed, prepared, changed_model_path, 11, changed_manifest)
    assert second != first
    assert OutputsLayout(changed.outputs_root).score_root(changed, 11) != OutputsLayout(
        config.outputs_root
    ).score_root(config, 11)
