from __future__ import annotations

from pathlib import Path

from fedcrg.paths import (
    ConfigLayout,
    ExperimentResultsBundleLayout,
    ModelCacheLayout,
    OutputsLayout,
    PreparedDatasetLayout,
    ScoreCacheLayout,
    experiment_results_root,
    prepared_dataset_family_root,
    prepared_dataset_root,
)
from fedcrg.types import DatasetId, ExperimentId


def test_config_layout_resolves_the_three_frozen_documents(tmp_path: Path) -> None:
    layout = ConfigLayout(tmp_path)
    assert layout.study == tmp_path / "study.yaml"
    assert layout.datasets == tmp_path / "datasets.yaml"
    assert layout.experiments == tmp_path / "experiments.yaml"


def test_prepared_dataset_layout_resolves_cache_artifacts(tmp_path: Path) -> None:
    layout = PreparedDatasetLayout(tmp_path)
    assert layout.manifest == tmp_path / "manifest.json"
    assert layout.preprocessing == tmp_path / "preprocessing.json"
    assert layout.eligibility == tmp_path / "eligibility.json"
    assert layout.diad_eligibility == tmp_path / "diad_eligibility.json"
    assert layout.raw_staging == tmp_path / "_raw"
    assert layout.splits == tmp_path / "splits"
    assert layout.seeded_splits == tmp_path / "splits" / "seeded"
    assert layout.source_order_split == tmp_path / "splits" / "source_order.json"


def test_prepared_dataset_root_propagates_custom_preprocessed_root(tmp_path: Path) -> None:
    root = prepared_dataset_root(tmp_path, DatasetId.NBAIOT, "a" * 64, "b" * 64)
    assert root == tmp_path / "nbaiot" / f"{'a' * 16}-{'b' * 16}"
    assert prepared_dataset_family_root(tmp_path, DatasetId.NBAIOT) == tmp_path / "nbaiot"


def test_outputs_layout_returns_model_and_score_cache_layouts(tmp_path: Path) -> None:
    from tests._fixtures import primary_experiment_config

    config = primary_experiment_config(tmp_path)
    assert config.detector is not None
    detector_id = config.detector.id
    layout = OutputsLayout(config.outputs_root)

    model_cache = layout.model_cache(config, 11)
    assert isinstance(model_cache, ModelCacheLayout)
    assert model_cache.root == (
        layout.cache_models
        / config.dataset.id
        / detector_id
        / "m11"
        / config.training_spec_hash[:16]
    )
    assert model_cache.model == model_cache.root / "model.pt"
    assert model_cache.training_manifest == model_cache.root / "training.json"

    score_cache = layout.score_cache(config, 11)
    assert isinstance(score_cache, ScoreCacheLayout)
    assert score_cache.root == (
        layout.cache_scores
        / config.dataset.id
        / detector_id
        / "m11"
        / config.training_spec_hash[:16]
    )
    assert score_cache.manifest == score_cache.root / "manifest.json"
    assert score_cache.score == score_cache.root / "score_cache.parquet"


def test_outputs_layout_run_and_analysis(tmp_path: Path) -> None:
    from fedcrg.types import ExperimentId

    layout = OutputsLayout(tmp_path)
    assert layout.run("run-1").root == layout.runs / "run-1"
    assert layout.analysis_result(ExperimentId.PRIMARY_NBAIOT) == (
        layout.cache_analysis / "primary_nbaiot.json"
    )


def test_outputs_layout_experiment_json_tables_and_figures(tmp_path: Path) -> None:
    from fedcrg.types import ExperimentId

    layout = OutputsLayout(tmp_path)
    assert layout.experiment_json(ExperimentId.PRIMARY_NBAIOT) == (layout.json / "primary_nbaiot")
    assert layout.experiment_tables(ExperimentId.PRIMARY_NBAIOT) == (
        layout.tables / "primary_nbaiot"
    )
    assert layout.experiment_figures(ExperimentId.PRIMARY_NBAIOT) == (
        layout.figures / "primary_nbaiot"
    )


def test_experiment_results_bundle_layout_required_directories_are_paths(tmp_path: Path) -> None:
    layout = ExperimentResultsBundleLayout(tmp_path)
    assert layout.required_directories == (
        layout.json_dir,
        layout.csv_dir,
        layout.figures,
        layout.provenance,
    )
    assert all(isinstance(path, Path) for path in layout.required_directories)


def test_experiment_results_root(tmp_path: Path) -> None:
    assert (
        experiment_results_root(tmp_path, ExperimentId.PRIMARY_NBAIOT)
        == tmp_path / "experiments" / "primary_nbaiot"
    )
