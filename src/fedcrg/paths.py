"""Filesystem layout: the single place that knows how repository and runtime
paths are constructed.

Business/scientific code asks a layout object for a path; it never
reconstructs a known repository path itself. This module only calculates
paths — it never reads, writes, hashes, or validates files, and it owns no
scientific logic.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from fedcrg.types import CampaignId, DatasetId, ExperimentId, ModelSeed, RunId, Sha256

if TYPE_CHECKING:
    from fedcrg.config import ExperimentConfig

FrozenModel = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)


class LayoutDirectory(StrEnum):
    """Reserved directory names of the immutable output layout."""

    OUTPUTS = "outputs"
    RUNS = "runs"
    DATA = "data"
    TRAINING = "training"
    SCORES = "scores"
    DECISIONS = "decisions"
    METRICS = "metrics"
    TABLES = "tables"
    FIGURES = "figures"
    REPORTS = "reports"
    LOGS = "logs"
    VERIFICATION = "verification"
    CACHE = "cache"
    MODELS = "models"
    ANALYSIS = "analysis"
    CAMPAIGNS = "campaigns"
    MONITORING = "monitoring"
    PUBLICATION = "publication"
    LATEST = "latest"
    STATISTICS = "statistics"
    PROVENANCE = "provenance"
    RESOLVED_CONFIGS = "resolved_configs"
    SPLITS = "splits"
    SEEDED = "seeded"


class LayoutArtifact(StrEnum):
    """Reserved artifact filenames of the immutable output layout."""

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
    PRIMARY_NBAIOT_CONFIG = "primary_nbaiot.json"
    METRIC_RECORDS_BUNDLE = "metric_records.json"
    READINESS_PLANS = "readiness_plans.json"
    MISMATCH_CUTOFFS = "mismatch_cutoffs.json"
    PROVENANCE = "provenance.json"
    TELEMETRY = "telemetry.jsonl"
    BENCHMARK = "benchmark.json"
    SCORE_CACHE = "score_cache.parquet"


class StudyPaths(BaseModel):
    """Repository layout roots; the CLI accepts no path options."""

    model_config = FrozenModel

    data_root: Path
    preprocessed_root: Path
    outputs_root: Path
    results_root: Path


class ConfigLayout:
    """Location of the three frozen configuration documents."""

    def __init__(self, config_root: Path = Path("config")) -> None:
        self.config_root = config_root

    @property
    def study(self) -> Path:
        return self.config_root / "study.yaml"

    @property
    def datasets(self) -> Path:
        return self.config_root / "datasets.yaml"

    @property
    def experiments(self) -> Path:
        return self.config_root / "experiments.yaml"


class PreparedDatasetLayout:
    """One prepared-cache root: manifests, eligibility, staging, and calibration splits."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def manifest(self) -> Path:
        return self.root / LayoutArtifact.MANIFEST.value

    @property
    def preprocessing(self) -> Path:
        return self.root / LayoutArtifact.PREPROCESSING.value

    @property
    def eligibility(self) -> Path:
        return self.root / LayoutArtifact.ELIGIBILITY.value

    @property
    def diad_eligibility(self) -> Path:
        return self.root / LayoutArtifact.DIAD_ELIGIBILITY.value

    @property
    def raw_staging(self) -> Path:
        return self.root / "_raw"

    @property
    def splits(self) -> Path:
        return self.root / LayoutDirectory.SPLITS.value

    @property
    def seeded_splits(self) -> Path:
        return self.splits / LayoutDirectory.SEEDED.value

    @property
    def source_order_split(self) -> Path:
        return self.splits / LayoutArtifact.SOURCE_ORDER_ASSIGNMENT.value


def prepared_dataset_family_root(preprocessed_root: Path, dataset_id: DatasetId) -> Path:
    """The prepared-cache root shared by every specification of one dataset."""
    return preprocessed_root / dataset_id.value


def prepared_dataset_root(
    preprocessed_root: Path,
    dataset_id: DatasetId,
    data_spec_hash: Sha256,
    source_identity_hash: Sha256,
) -> Path:
    """The immutable prepared-cache root for one data specification and source identity."""
    return (
        prepared_dataset_family_root(preprocessed_root, dataset_id)
        / f"{data_spec_hash[:16]}-{source_identity_hash[:16]}"
    )


class ModelCacheLayout:
    """One frozen-model cache root: the model file and its training manifest."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def model(self) -> Path:
        return self.root / LayoutArtifact.MODEL.value

    @property
    def training_manifest(self) -> Path:
        return self.root / LayoutArtifact.TRAINING.value


class ScoreCacheLayout:
    """One frozen-score cache root: the manifest and the score-cache artifact."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def manifest(self) -> Path:
        return self.root / LayoutArtifact.MANIFEST.value

    @property
    def score(self) -> Path:
        return self.root / LayoutArtifact.SCORE_CACHE.value


class RunLayout:
    """Immutable per-run output directory layout."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def manifest(self) -> Path:
        return self.root / LayoutArtifact.MANIFEST.value

    @property
    def run_config(self) -> Path:
        return self.root / LayoutArtifact.RUN_CONFIG.value

    @property
    def resolved_config(self) -> Path:
        return self.root / LayoutArtifact.RESOLVED_CONFIG.value

    @property
    def environment(self) -> Path:
        return self.root / LayoutArtifact.ENVIRONMENT.value

    @property
    def data(self) -> Path:
        return self.root / LayoutDirectory.DATA.value

    @property
    def training(self) -> Path:
        return self.root / LayoutDirectory.TRAINING.value

    @property
    def model_reference(self) -> Path:
        return self.training / LayoutArtifact.MODEL_REFERENCE.value

    @property
    def scores(self) -> Path:
        return self.root / LayoutDirectory.SCORES.value

    @property
    def score_reference(self) -> Path:
        return self.scores / LayoutArtifact.SCORE_REFERENCE.value

    @property
    def decisions(self) -> Path:
        return self.root / LayoutDirectory.DECISIONS.value

    @property
    def threshold_records(self) -> Path:
        return self.decisions / LayoutArtifact.THRESHOLD_RECORDS.value

    @property
    def metrics(self) -> Path:
        return self.root / LayoutDirectory.METRICS.value

    @property
    def metric_records(self) -> Path:
        return self.metrics / LayoutArtifact.METRIC_RECORDS.value

    @property
    def federation_metrics(self) -> Path:
        return self.metrics / LayoutArtifact.FEDERATION.value

    @property
    def tables(self) -> Path:
        return self.root / LayoutDirectory.TABLES.value

    @property
    def figures(self) -> Path:
        return self.root / LayoutDirectory.FIGURES.value

    @property
    def reports(self) -> Path:
        return self.root / LayoutDirectory.REPORTS.value

    @property
    def admission(self) -> Path:
        return self.metrics / LayoutArtifact.ADMISSION.value

    @property
    def evaluation_summary(self) -> Path:
        return self.reports / LayoutArtifact.EVALUATION_SUMMARY.value

    @property
    def dataset_manifest(self) -> Path:
        return self.data / LayoutArtifact.DATASET_MANIFEST.value

    @property
    def preprocessing_evidence(self) -> Path:
        return self.data / LayoutArtifact.PREPROCESSING.value

    @property
    def eligibility_manifest(self) -> Path:
        return self.data / LayoutArtifact.ELIGIBILITY.value

    @property
    def split_manifest(self) -> Path:
        return self.data / LayoutArtifact.CALIBRATION_ASSIGNMENT.value

    @property
    def training_manifest(self) -> Path:
        return self.training / LayoutArtifact.TRAINING.value

    @property
    def score_manifest(self) -> Path:
        return self.scores / LayoutArtifact.MANIFEST.value

    @property
    def logs(self) -> Path:
        return self.root / LayoutDirectory.LOGS.value

    @property
    def verification(self) -> Path:
        return self.root / LayoutDirectory.VERIFICATION.value

    @property
    def hashes(self) -> Path:
        return self.verification / LayoutArtifact.HASHES.value

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
            self.reports,
            self.logs,
            self.verification,
        ):
            directory.mkdir()


class PublicationLayout:
    """One publication package root: tables, figures, and its manifest."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def tables(self) -> Path:
        return self.root / LayoutDirectory.TABLES.value

    @property
    def figures(self) -> Path:
        return self.root / LayoutDirectory.FIGURES.value

    @property
    def manifest(self) -> Path:
        return self.root / LayoutArtifact.MANIFEST.value


class ResultsBundleLayout:
    """Reserved publication-bundle artifact names under results/<campaign-id>/."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def manifest(self) -> Path:
        return self.root / LayoutArtifact.MANIFEST.value

    @property
    def checksums(self) -> Path:
        return self.root / LayoutArtifact.CHECKSUMS.value

    @property
    def resolved_configs(self) -> Path:
        return self.root / LayoutDirectory.RESOLVED_CONFIGS.value

    @property
    def metrics(self) -> Path:
        return self.root / LayoutDirectory.METRICS.value

    @property
    def statistics(self) -> Path:
        return self.root / LayoutDirectory.STATISTICS.value

    @property
    def tables(self) -> Path:
        return self.root / LayoutDirectory.TABLES.value

    @property
    def figures(self) -> Path:
        return self.root / LayoutDirectory.FIGURES.value

    @property
    def reports(self) -> Path:
        return self.root / LayoutDirectory.REPORTS.value

    @property
    def provenance(self) -> Path:
        return self.root / LayoutDirectory.PROVENANCE.value

    @property
    def primary_nbaiot_config(self) -> Path:
        return self.resolved_configs / LayoutArtifact.PRIMARY_NBAIOT_CONFIG.value

    @property
    def metric_records(self) -> Path:
        return self.metrics / LayoutArtifact.METRIC_RECORDS_BUNDLE.value

    @property
    def readiness_plans(self) -> Path:
        return self.statistics / LayoutArtifact.READINESS_PLANS.value

    @property
    def mismatch_cutoffs(self) -> Path:
        return self.statistics / LayoutArtifact.MISMATCH_CUTOFFS.value

    @property
    def provenance_json(self) -> Path:
        return self.provenance / LayoutArtifact.PROVENANCE.value

    @property
    def required_directories(self) -> tuple[Path, ...]:
        return (
            self.metrics,
            self.statistics,
            self.tables,
            self.figures,
            self.reports,
            self.provenance,
            self.resolved_configs,
        )


def campaign_results_root(results_root: Path, campaign_id: CampaignId) -> Path:
    """The immutable results-bundle root for one campaign."""
    return results_root / str(campaign_id)


def campaign_status_path(campaigns_root: Path, campaign_id: CampaignId) -> Path:
    """The persisted campaign-status snapshot path for one campaign."""
    value = str(campaign_id)
    if not value or "/" in value or ".." in value:
        raise ValueError(f"Invalid campaign id: {value!r}")
    return campaigns_root / f"{value}.json"


class OutputsLayout:
    """Reserved outputs/ directory tree: runs, caches, campaigns, logs,
    monitoring, reports, environment and telemetry files."""

    def __init__(self, outputs_root: Path = Path(LayoutDirectory.OUTPUTS.value)) -> None:
        self.outputs_root = outputs_root

    @property
    def runs(self) -> Path:
        return self.outputs_root / LayoutDirectory.RUNS.value

    @property
    def cache(self) -> Path:
        return self.outputs_root / LayoutDirectory.CACHE.value

    @property
    def cache_models(self) -> Path:
        return self.cache / LayoutDirectory.MODELS.value

    @property
    def cache_scores(self) -> Path:
        return self.cache / LayoutDirectory.SCORES.value

    @property
    def cache_analysis(self) -> Path:
        return self.cache / LayoutDirectory.ANALYSIS.value

    @property
    def campaigns(self) -> Path:
        return self.outputs_root / LayoutDirectory.CAMPAIGNS.value

    @property
    def logs(self) -> Path:
        return self.outputs_root / LayoutDirectory.LOGS.value

    @property
    def monitoring(self) -> Path:
        return self.outputs_root / LayoutDirectory.MONITORING.value

    @property
    def reports(self) -> Path:
        return self.outputs_root / LayoutDirectory.REPORTS.value

    @property
    def publication(self) -> PublicationLayout:
        return PublicationLayout(self.reports / LayoutDirectory.PUBLICATION.value)

    @property
    def environment_file(self) -> Path:
        return self.outputs_root / LayoutArtifact.ENVIRONMENT.value

    @property
    def telemetry_file(self) -> Path:
        return self.monitoring / LayoutArtifact.TELEMETRY.value

    @property
    def benchmark_report(self) -> Path:
        return self.reports / LayoutArtifact.BENCHMARK.value

    @property
    def readiness_plans_file(self) -> Path:
        return self.cache_analysis / LayoutArtifact.READINESS_PLANS.value

    @property
    def mismatch_cutoffs_file(self) -> Path:
        return self.cache_analysis / LayoutArtifact.MISMATCH_CUTOFFS.value

    def run(self, run_id: RunId) -> RunLayout:
        return RunLayout(self.runs / str(run_id))

    def analysis_result(self, experiment_id: ExperimentId) -> Path:
        return self.cache_analysis / f"{experiment_id.value}.json"

    def campaign_status(self, campaign_id: CampaignId) -> Path:
        return campaign_status_path(self.campaigns, campaign_id)

    def model_cache(self, config: ExperimentConfig, model_seed: ModelSeed) -> ModelCacheLayout:
        if config.detector is None:
            raise ValueError("Model cache requires a detector profile")
        return ModelCacheLayout(
            self.cache_models
            / config.dataset.id.value
            / config.detector.id.value
            / f"m{int(model_seed)}"
            / config.training_spec_hash[:16]
        )

    def score_cache(self, config: ExperimentConfig, model_seed: ModelSeed) -> ScoreCacheLayout:
        if config.detector is None:
            raise ValueError("Score cache requires a detector profile")
        return ScoreCacheLayout(
            self.cache_scores
            / config.dataset.id.value
            / config.detector.id.value
            / f"m{int(model_seed)}"
            / config.training_spec_hash[:16]
        )
