from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from fedcrg.types import DatasetId, ExperimentId, ModelSeed, RunId, Sha256

if TYPE_CHECKING:
    from fedcrg.config import ExperimentConfig

FrozenModel = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)


class LayoutDirectory(StrEnum):
    OUTPUTS = "outputs"
    RUNS = "runs"
    DATA = "data"
    TRAINING = "training"
    SCORES = "scores"
    DECISIONS = "decisions"
    METRICS = "metrics"
    TABLES = "tables"
    FIGURES = "figures"
    LOGS = "logs"
    VERIFICATION = "verification"
    CACHE = "cache"
    MODELS = "models"
    ANALYSIS = "analysis"
    MONITORING = "monitoring"
    STATISTICS = "statistics"
    PROVENANCE = "provenance"
    RESOLVED_CONFIGS = "resolved_configs"
    SPLITS = "splits"
    SEEDED = "seeded"
    CONFIG = "config"
    RAW_STAGING = "_raw"
    EXPERIMENTS = "experiments"
    JSON = "json"
    CSV = "csv"


class LayoutArtifact(StrEnum):
    MANIFEST = "manifest.json"
    RUN_CONFIG = "run_config.json"
    RESOLVED_CONFIG = "resolved_config.yaml"
    ENVIRONMENT = "environment.json"
    MODEL_REFERENCE = "model_reference.json"
    SCORE_REFERENCE = "cache_reference.json"
    THRESHOLD_RECORDS = "threshold_record.jsonl"
    METRIC_RECORDS = "metric_record.jsonl"
    FEDERATION = "federation.json"
    ADMISSION = "admission.json"
    EVALUATION_SUMMARY = "evaluation_summary.json"
    DATASET_MANIFEST = "dataset_manifest.json"
    PREPROCESSING = "preprocessing.json"
    ELIGIBILITY = "eligibility.json"
    DIAD_ELIGIBILITY = "diad_eligibility.json"
    CALIBRATION_ASSIGNMENT = "calibration_assignment.json"
    SOURCE_ORDER_ASSIGNMENT = "source_order.json"
    TRAINING = "training.json"
    MODEL = "model.pt"
    HASHES = "hashes.json"
    CHECKSUMS = "checksums.json"
    METRIC_RECORDS_BUNDLE = "metric_records.json"
    READINESS_PLANS = "readiness_plans.json"
    MISMATCH_CUTOFFS = "mismatch_cutoffs.json"
    PROVENANCE = "provenance.json"
    TELEMETRY = "telemetry.jsonl"
    SCORE_CACHE = "score_cache.parquet"
    EXPERIMENT_RESULT_JSON = "results.json"
    EXPERIMENT_CELLS_CSV = "cells.csv"
    EXPERIMENT_POLICY_CSV = "policy_metrics.csv"
    EXPERIMENT_BENCHMARK_CSV = "benchmark.csv"
    EXPERIMENT_COVERAGE_FIGURE = "coverage.png"
    EXPERIMENT_POWER_FIGURE = "power.png"
    EXPERIMENT_OPERATING_POINTS_FIGURE = "operating_points.png"
    EXPERIMENT_RELIABILITY_FIGURE = "reliability_utility.png"
    EXPERIMENT_REPLICATION_FIGURE = "external_replication.png"
    EMPIRICAL_RESULTS = "empirical_results.json"


class ConfigArtifact(StrEnum):
    STUDY = "study.yaml"
    DATASETS = "datasets.yaml"
    EXPERIMENTS = "experiments.yaml"


class StudyPaths(BaseModel):
    model_config = FrozenModel

    data_root: Path
    preprocessed_root: Path
    outputs_root: Path
    results_root: Path


class ConfigLayout:
    def __init__(self, config_root: Path = Path(LayoutDirectory.CONFIG)) -> None:
        self.config_root = config_root

    @property
    def study(self) -> Path:
        return self.config_root / ConfigArtifact.STUDY

    @property
    def datasets(self) -> Path:
        return self.config_root / ConfigArtifact.DATASETS

    @property
    def experiments(self) -> Path:
        return self.config_root / ConfigArtifact.EXPERIMENTS


class PreparedDatasetLayout:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def manifest(self) -> Path:
        return self.root / LayoutArtifact.MANIFEST

    @property
    def preprocessing(self) -> Path:
        return self.root / LayoutArtifact.PREPROCESSING

    @property
    def eligibility(self) -> Path:
        return self.root / LayoutArtifact.ELIGIBILITY

    @property
    def diad_eligibility(self) -> Path:
        return self.root / LayoutArtifact.DIAD_ELIGIBILITY

    @property
    def raw_staging(self) -> Path:
        return self.root / LayoutDirectory.RAW_STAGING

    @property
    def splits(self) -> Path:
        return self.root / LayoutDirectory.SPLITS

    @property
    def seeded_splits(self) -> Path:
        return self.splits / LayoutDirectory.SEEDED

    @property
    def source_order_split(self) -> Path:
        return self.splits / LayoutArtifact.SOURCE_ORDER_ASSIGNMENT


def prepared_dataset_family_root(preprocessed_root: Path, dataset_id: DatasetId) -> Path:
    return preprocessed_root / dataset_id


def prepared_dataset_root(
    preprocessed_root: Path,
    dataset_id: DatasetId,
    data_spec_hash: Sha256,
    source_identity_hash: Sha256,
) -> Path:
    return (
        prepared_dataset_family_root(preprocessed_root, dataset_id)
        / f"{data_spec_hash[:16]}-{source_identity_hash[:16]}"
    )


class ModelCacheLayout:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def model(self) -> Path:
        return self.root / LayoutArtifact.MODEL

    @property
    def training_manifest(self) -> Path:
        return self.root / LayoutArtifact.TRAINING


class ScoreCacheLayout:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def manifest(self) -> Path:
        return self.root / LayoutArtifact.MANIFEST

    @property
    def score(self) -> Path:
        return self.root / LayoutArtifact.SCORE_CACHE


class RunLayout:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def manifest(self) -> Path:
        return self.root / LayoutArtifact.MANIFEST

    @property
    def run_config(self) -> Path:
        return self.root / LayoutArtifact.RUN_CONFIG

    @property
    def resolved_config(self) -> Path:
        return self.root / LayoutArtifact.RESOLVED_CONFIG

    @property
    def environment(self) -> Path:
        return self.root / LayoutArtifact.ENVIRONMENT

    @property
    def data(self) -> Path:
        return self.root / LayoutDirectory.DATA

    @property
    def training(self) -> Path:
        return self.root / LayoutDirectory.TRAINING

    @property
    def model_reference(self) -> Path:
        return self.training / LayoutArtifact.MODEL_REFERENCE

    @property
    def scores(self) -> Path:
        return self.root / LayoutDirectory.SCORES

    @property
    def score_reference(self) -> Path:
        return self.scores / LayoutArtifact.SCORE_REFERENCE

    @property
    def decisions(self) -> Path:
        return self.root / LayoutDirectory.DECISIONS

    @property
    def threshold_records(self) -> Path:
        return self.decisions / LayoutArtifact.THRESHOLD_RECORDS

    @property
    def metrics(self) -> Path:
        return self.root / LayoutDirectory.METRICS

    @property
    def metric_records(self) -> Path:
        return self.metrics / LayoutArtifact.METRIC_RECORDS

    @property
    def federation_metrics(self) -> Path:
        return self.metrics / LayoutArtifact.FEDERATION

    @property
    def tables(self) -> Path:
        return self.root / LayoutDirectory.TABLES

    @property
    def figures(self) -> Path:
        return self.root / LayoutDirectory.FIGURES

    @property
    def admission(self) -> Path:
        return self.metrics / LayoutArtifact.ADMISSION

    @property
    def evaluation_summary(self) -> Path:
        return self.metrics / LayoutArtifact.EVALUATION_SUMMARY

    @property
    def dataset_manifest(self) -> Path:
        return self.data / LayoutArtifact.DATASET_MANIFEST

    @property
    def preprocessing_evidence(self) -> Path:
        return self.data / LayoutArtifact.PREPROCESSING

    @property
    def eligibility_manifest(self) -> Path:
        return self.data / LayoutArtifact.ELIGIBILITY

    @property
    def split_manifest(self) -> Path:
        return self.data / LayoutArtifact.CALIBRATION_ASSIGNMENT

    @property
    def training_manifest(self) -> Path:
        return self.training / LayoutArtifact.TRAINING

    @property
    def score_manifest(self) -> Path:
        return self.scores / LayoutArtifact.MANIFEST

    @property
    def logs(self) -> Path:
        return self.root / LayoutDirectory.LOGS

    @property
    def verification(self) -> Path:
        return self.root / LayoutDirectory.VERIFICATION

    @property
    def hashes(self) -> Path:
        return self.verification / LayoutArtifact.HASHES

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        for directory in (
            self.data,
            self.training,
            self.scores,
            self.decisions,
            self.metrics,
            self.tables,
            self.figures,
            self.logs,
            self.verification,
        ):
            directory.mkdir()


class ExperimentResultsBundleLayout:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def manifest(self) -> Path:
        return self.root / LayoutArtifact.MANIFEST

    @property
    def checksums(self) -> Path:
        return self.root / LayoutArtifact.CHECKSUMS

    @property
    def json_dir(self) -> Path:
        return self.root / LayoutDirectory.JSON

    @property
    def csv_dir(self) -> Path:
        return self.root / LayoutDirectory.CSV

    @property
    def figures(self) -> Path:
        return self.root / LayoutDirectory.FIGURES

    @property
    def provenance(self) -> Path:
        return self.root / LayoutDirectory.PROVENANCE

    @property
    def resolved_configs(self) -> Path:
        return self.provenance / LayoutDirectory.RESOLVED_CONFIGS

    @property
    def metrics(self) -> Path:
        return self.json_dir

    @property
    def metric_records(self) -> Path:
        return self.json_dir / LayoutArtifact.METRIC_RECORDS_BUNDLE

    @property
    def analysis(self) -> Path:
        return self.json_dir

    @property
    def result_json(self) -> Path:
        return self.json_dir / LayoutArtifact.EXPERIMENT_RESULT_JSON

    @property
    def provenance_json(self) -> Path:
        return self.provenance / LayoutArtifact.PROVENANCE

    @property
    def runs(self) -> Path:
        return self.root / LayoutDirectory.RUNS

    @property
    def required_directories(self) -> tuple[Path, ...]:
        return (
            self.json_dir,
            self.csv_dir,
            self.figures,
            self.provenance,
        )


def experiment_results_root(results_root: Path, experiment_id: ExperimentId) -> Path:
    return results_root / LayoutDirectory.EXPERIMENTS / str(experiment_id)


class OutputsLayout:
    def __init__(self, outputs_root: Path = Path(LayoutDirectory.OUTPUTS)) -> None:
        self.outputs_root = outputs_root

    @property
    def runs(self) -> Path:
        return self.outputs_root / LayoutDirectory.RUNS

    @property
    def cache(self) -> Path:
        return self.outputs_root / LayoutDirectory.CACHE

    @property
    def cache_models(self) -> Path:
        return self.cache / LayoutDirectory.MODELS

    @property
    def cache_scores(self) -> Path:
        return self.cache / LayoutDirectory.SCORES

    @property
    def cache_analysis(self) -> Path:
        return self.cache / LayoutDirectory.ANALYSIS

    @property
    def logs(self) -> Path:
        return self.outputs_root / LayoutDirectory.LOGS

    @property
    def monitoring(self) -> Path:
        return self.outputs_root / LayoutDirectory.MONITORING

    @property
    def json(self) -> Path:
        return self.outputs_root / LayoutDirectory.JSON

    @property
    def figures(self) -> Path:
        return self.outputs_root / LayoutDirectory.FIGURES

    @property
    def tables(self) -> Path:
        return self.outputs_root / LayoutDirectory.TABLES

    def experiment_json(self, experiment_id: ExperimentId) -> Path:
        return self.json / str(experiment_id)

    def experiment_figures(self, experiment_id: ExperimentId) -> Path:
        return self.figures / str(experiment_id)

    def experiment_tables(self, experiment_id: ExperimentId) -> Path:
        return self.tables / str(experiment_id)

    @property
    def environment_file(self) -> Path:
        return self.outputs_root / LayoutArtifact.ENVIRONMENT

    @property
    def telemetry_file(self) -> Path:
        return self.monitoring / LayoutArtifact.TELEMETRY

    @property
    def readiness_plans_file(self) -> Path:
        return self.cache_analysis / LayoutArtifact.READINESS_PLANS

    @property
    def mismatch_cutoffs_file(self) -> Path:
        return self.cache_analysis / LayoutArtifact.MISMATCH_CUTOFFS

    def run(self, run_id: RunId) -> RunLayout:
        return RunLayout(self.runs / str(run_id))

    def analysis_result(self, experiment_id: ExperimentId) -> Path:
        return self.cache_analysis / f"{experiment_id}.json"

    def model_cache(self, config: ExperimentConfig, model_seed: ModelSeed) -> ModelCacheLayout:
        if config.detector is None:
            raise ValueError("Model cache requires a detector profile")
        return ModelCacheLayout(
            self.cache_models
            / config.dataset.id
            / config.detector.id
            / f"m{int(model_seed)}"
            / config.training_spec_hash[:16]
        )

    def score_cache(self, config: ExperimentConfig, model_seed: ModelSeed) -> ScoreCacheLayout:
        if config.detector is None:
            raise ValueError("Score cache requires a detector profile")
        return ScoreCacheLayout(
            self.cache_scores
            / config.dataset.id
            / config.detector.id
            / f"m{int(model_seed)}"
            / config.training_spec_hash[:16]
        )
