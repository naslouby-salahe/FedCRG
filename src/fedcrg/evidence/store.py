"""Evidence persistence: immutable output layout, run identities, atomic
JSON/JSONL writes, file hashing, and manifest stores."""

from __future__ import annotations

import json
import os
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Generic, TypeVar

import yaml
from pydantic import BaseModel

from fedcrg.config import ExperimentConfig, ExperimentSpec
from fedcrg.evidence.models import (
    CalibrationAssignmentManifest,
    CacheReference,
    EligibilityManifest,
    EnvironmentPin,
    GitEnvironment,
    PreparedDatasetManifest,
    RunManifest,
    TrainingManifest,
)
from fedcrg.hashing import sha256_file, sha256_text
from fedcrg.types import (
    ArtifactType,
    CalibrationSeed,
    DetectorId,
    ExperimentStatus,
    Identifier,
    JsonValue,
    ModelSeed,
    PathString,
    PolicyId,
    RunId,
    Sha256,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


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
    CALIBRATION_ASSIGNMENT = "calibration_assignment.json"
    TRAINING = "training.json"
    HASHES = "hashes.json"
    CHECKSUMS = "checksums.json"
    PRIMARY_NBAIOT_CONFIG = "primary_nbaiot.json"
    METRIC_RECORDS_BUNDLE = "metric_records.json"
    READINESS_PLANS = "readiness_plans.json"
    MISMATCH_CUTOFFS = "mismatch_cutoffs.json"
    PROVENANCE = "provenance.json"
    TELEMETRY = "telemetry.jsonl"
    BENCHMARK = "benchmark.json"


def _jsonable(value: object) -> JsonValue:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_write_json(path: Path, payload: object) -> None:
    """Atomically persist one JSON-serializable payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically persist one text payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def write_jsonl(path: Path, records: tuple[BaseModel, ...]) -> None:
    """Atomically append one JSON-lines document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
    os.replace(temp, path)


def load_json_model(path: Path, model: type[ModelT]) -> ModelT:
    """Load and validate one pydantic model from JSON."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    return model.model_validate(raw)


def load_yaml_mapping(path: Path) -> object:
    """Load one configuration document before validation."""
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration document must be a mapping: {path}")
    return {str(key): value for key, value in raw.items()}


class ModelStore(Generic[ModelT]):
    """Generic atomic JSON persistence for one frozen pydantic schema."""

    model: type[ModelT]

    def save(self, path: Path, manifest: ModelT) -> None:
        atomic_write_json(path, manifest)

    def load(self, path: Path) -> ModelT:
        return load_json_model(path, self.model)

    def load_model(self, path: Path) -> ModelT:
        return load_json_model(path, self.model)


class PreparedDatasetManifestStore(ModelStore[PreparedDatasetManifest]):
    """Atomic store for prepared-dataset manifests."""

    model = PreparedDatasetManifest


class TrainingManifestStore(ModelStore[TrainingManifest]):
    """Atomic store for training manifests."""

    model = TrainingManifest


class RunManifestStore(ModelStore[RunManifest]):
    """Atomic store for run manifests.

    Completed and failed runs are immutable: once a manifest is recorded as
    COMPLETE or FAILED it cannot be overwritten by another status, so run
    evidence cannot silently change after finalization.
    """

    model = RunManifest

    _TERMINAL_STATUSES = frozenset({ExperimentStatus.COMPLETE.value, ExperimentStatus.FAILED.value})

    def save(self, path: Path, manifest: RunManifest) -> None:
        if path.is_file():
            try:
                existing = self.load(path)
            except Exception:
                existing = None
            if (
                existing is not None
                and existing.status.value in self._TERMINAL_STATUSES
                and existing.status is not manifest.status
            ):
                from fedcrg.types import ImmutableRunError

                raise ImmutableRunError(
                    f"Cannot overwrite a finalized run manifest: {path} "
                    f"({existing.status.value} -> {manifest.status.value})"
                )
        super().save(path, manifest)


class EligibilityManifestStore(ModelStore[EligibilityManifest]):
    """Atomic store for eligibility manifests."""

    model = EligibilityManifest


class CalibrationAssignmentManifestStore(ModelStore[CalibrationAssignmentManifest]):
    """Atomic store for calibration-assignment manifests."""

    model = CalibrationAssignmentManifest


class CacheReferenceStore(ModelStore[CacheReference]):
    """Atomic store for cache references."""

    model = CacheReference

    @staticmethod
    def build(path: Path, outputs_root: Path) -> CacheReference:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(outputs_root.resolve()).as_posix()
        except ValueError:
            relative = resolved.as_posix()
        return CacheReference(
            relative_path=relative,
            sha256=sha256_file(path),
            size_bytes=int(path.stat().st_size),
        )


class RunLayout:
    """Immutable per-run output directory layout."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def for_run(cls, outputs_root: Path, run_id: RunId) -> RunLayout:
        return cls(OutputsLayout(outputs_root).runs / str(run_id))

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


class PreparedLayout:
    """Reserved prepared-cache artifact names: manifests, preprocessing and
    eligibility evidence, training manifests, model files, raw staging, and
    calibration-split files."""

    manifest_filename = "manifest.json"
    preprocessing_filename = "preprocessing.json"
    eligibility_filename = "eligibility.json"
    diad_eligibility_filename = "diad_eligibility.json"
    training_filename = "training.json"
    model_filename = "model.pt"
    raw_staging_directory = "_raw"
    calibration_split_directory = "splits/seeded"
    source_order_split_filename = "splits/source_order.json"


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
    def publication_root(self) -> Path:
        return self.reports / LayoutDirectory.PUBLICATION.value

    @property
    def publication_manifest(self) -> Path:
        return self.publication_root / LayoutArtifact.MANIFEST.value

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

    def model_root(
        self,
        config: ExperimentConfig,
        model_seed: ModelSeed,
    ) -> Path:
        if config.detector is None:
            raise ValueError("Model cache requires a detector profile")
        return (
            self.cache_models
            / config.dataset.id.value
            / config.detector.id.value
            / f"m{int(model_seed)}"
            / config.training_spec_hash[:16]
        )

    def score_root(
        self,
        config: ExperimentConfig,
        model_seed: ModelSeed,
    ) -> Path:
        if config.detector is None:
            raise ValueError("Score cache requires a detector profile")
        return (
            self.cache_scores
            / config.dataset.id.value
            / config.detector.id.value
            / f"m{int(model_seed)}"
            / config.training_spec_hash[:16]
        )


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
    def required_directories(self) -> tuple[Identifier, ...]:
        return tuple(
            directory.value
            for directory in (
                LayoutDirectory.METRICS,
                LayoutDirectory.STATISTICS,
                LayoutDirectory.TABLES,
                LayoutDirectory.FIGURES,
                LayoutDirectory.REPORTS,
                LayoutDirectory.PROVENANCE,
                LayoutDirectory.RESOLVED_CONFIGS,
            )
        )


def build_run_id(
    config: ExperimentConfig,
    model_seed: ModelSeed,
    calibration_seed: CalibrationSeed,
    policy: PolicyId,
) -> RunId:
    """Build a path-safe run id from one fully resolved scientific configuration."""
    protocol = config.protocol
    detector = config.detector
    if detector is None:
        raise ValueError("Run identity requires a real-data detector profile")
    alpha_ppm = round(protocol.alpha * 1_000_000)
    rho_bp = round(protocol.rho * 10_000)
    assurance_bp = round(protocol.readiness_assurance * 10_000)
    confidence_bp = round(protocol.mismatch_confidence * 10_000)
    detector_label = "ae" if detector.id is DetectorId.AUTOENCODER else detector.id.value
    prefix = (
        f"{config.dataset.id.value}__{detector_label}__ms{int(model_seed)}__"
        f"cs{int(calibration_seed)}__a{alpha_ppm}__r{rho_bp}__"
        f"ga{assurance_bp}__gb{confidence_bp}__{policy.value.lower()}"
    )
    return f"{prefix}__cfg{config.config_hash[:12]}"


class FileHashRecord:
    """One file's relative path and SHA-256."""

    def __init__(self, relative_path: Identifier, sha256: Sha256) -> None:
        self.relative_path = relative_path
        self.sha256 = sha256


class VerificationResult:
    """Outcome of an artifact verification audit."""

    def __init__(
        self,
        valid: bool,
        missing: tuple[Identifier, ...],
        mismatched: tuple[Identifier, ...],
        hashes: tuple[FileHashRecord, ...],
    ) -> None:
        self.valid = valid
        self.missing = missing
        self.mismatched = mismatched
        self.hashes = hashes

    def hash_for(self, relative_path: Identifier) -> Sha256 | None:
        for record in self.hashes:
            if record.relative_path == relative_path:
                return record.sha256
        return None


class ArtifactVerifier:
    """Verify file hashes and the provenance chain across run evidence."""

    def _path_for(self, layout: RunLayout, artifact: ArtifactType) -> Path | None:
        mapping = {
            ArtifactType.RESOLVED_CONFIG: layout.resolved_config,
            ArtifactType.DATASET_MANIFEST: layout.dataset_manifest,
            ArtifactType.ELIGIBILITY_MANIFEST: layout.data / LayoutArtifact.ELIGIBILITY.value,
            ArtifactType.SPLIT_MANIFEST: layout.data / LayoutArtifact.CALIBRATION_ASSIGNMENT.value,
            ArtifactType.PREPROCESSING_MANIFEST: layout.preprocessing_evidence,
            ArtifactType.TRAINING_MANIFEST: layout.training_manifest,
            ArtifactType.MODEL: layout.model_reference,
            ArtifactType.SCORE_MANIFEST: layout.score_manifest,
            ArtifactType.THRESHOLD_RECORDS: layout.threshold_records,
            ArtifactType.METRICS: layout.metric_records,
            ArtifactType.VERIFICATION: layout.hashes,
        }
        return mapping.get(artifact)

    def required_files(
        self,
        layout: RunLayout,
        definition: ExperimentSpec,
    ) -> tuple[Path, ...]:
        required = [
            layout.run_config,
            layout.resolved_config,
            layout.environment,
            layout.manifest,
        ]
        for artifact in definition.required_evidence:
            path = self._path_for(layout, artifact)
            if path is None:
                continue
            required.append(path)
        return tuple(required)

    def record(self, layout: RunLayout, definition: ExperimentSpec) -> VerificationResult:
        missing: list[PathString] = []
        mismatched: list[PathString] = []
        hashes: list[FileHashRecord] = []
        for path in self.required_files(layout, definition):
            relative = path.relative_to(layout.root).as_posix()
            if not path.is_file():
                missing.append(relative)
                continue
            digest = sha256_file(path)
            hashes.append(FileHashRecord(relative, digest))
            expected = self._expected_hash(layout, relative)
            if expected is not None and digest != expected:
                mismatched.append(relative)
        if layout.verification.is_dir():
            atomic_write_json(
                layout.hashes,
                {
                    "files": [
                        {"relative_path": item.relative_path, "sha256": item.sha256}
                        for item in hashes
                    ]
                },
            )
        return VerificationResult(
            valid=not missing and not mismatched,
            missing=tuple(missing),
            mismatched=tuple(mismatched),
            hashes=tuple(hashes),
        )

    @staticmethod
    def _expected_hash(layout: RunLayout, relative: Identifier) -> Sha256 | None:
        reference_paths = {
            layout.model_reference.relative_to(layout.root).as_posix(): layout.model_reference,
            layout.score_reference.relative_to(layout.root).as_posix(): layout.score_reference,
        }
        reference = reference_paths.get(relative)
        if reference is None or not reference.is_file():
            return None
        try:
            record = CacheReference.model_validate(
                json.loads(reference.read_text(encoding="utf-8"))
            )
        except Exception:
            return None
        return record.sha256


def capture_environment(repository_root: Path) -> GitEnvironment:
    """Freeze repository and Python environment evidence."""
    import subprocess

    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(repository_root),
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    clean = not status
    patch = run("diff")
    patch_hash: Sha256 | None = None
    if patch:
        patch_hash = sha256_text(patch)
    import platform
    import sys

    import torch

    pin = EnvironmentPin(
        python=sys.version.split()[0],
        torch=torch.__version__,
        platform=platform.platform(),
        commit=commit,
        patch_sha256=patch_hash,
    )
    return GitEnvironment(
        git_commit=commit,
        git_clean=clean,
        git_patch_sha256=patch_hash,
        environment_pin_sha256=pin.sha256,
        environment_pin_kind="python-torch-platform-commit",
    )
