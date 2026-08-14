"""Evidence persistence: immutable output layout, run identities, atomic
JSON/JSONL writes, file hashing, and manifest stores."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Generic, TypeVar

import yaml
from pydantic import BaseModel

from fedcrg.config import ExperimentConfig
from fedcrg.evidence.models import (
    CalibrationAssignmentManifest,
    CacheReference,
    EligibilityManifest,
    GitEnvironment,
    PreparedDatasetManifest,
    RunManifest,
    TrainingManifest,
)
from fedcrg.config import ExperimentSpec
from fedcrg.types import (
    ArtifactType,
    ByteCount,
    CalibrationSeed,
    DetectorId,
    ExperimentStatus,
    Identifier,
    JsonValue,
    ModelSeed,
    PolicyId,
    RunId,
    Sha256,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def sha256_file(path: Path, chunk_size: ByteCount = 1024 * 1024) -> Sha256:
    """SHA-256 of one file using an IO-sized read chunk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


class PreparedDatasetManifestStore(ModelStore):
    """Atomic store for prepared-dataset manifests."""
    model = PreparedDatasetManifest


class TrainingManifestStore(ModelStore):
    """Atomic store for training manifests."""
    model = TrainingManifest


class RunManifestStore(ModelStore):
    """Atomic store for run manifests."""
    model = RunManifest


class EligibilityManifestStore(ModelStore):
    """Atomic store for eligibility manifests."""
    model = EligibilityManifest


class CalibrationAssignmentManifestStore(ModelStore):
    """Atomic store for calibration-assignment manifests."""
    model = CalibrationAssignmentManifest


class CacheReferenceStore(ModelStore):
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
        return cls(outputs_root / "runs" / str(run_id))

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def run_config(self) -> Path:
        return self.root / "run_config.json"

    @property
    def resolved_config(self) -> Path:
        return self.root / "resolved_config.yaml"

    @property
    def environment(self) -> Path:
        return self.root / "environment.json"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def training(self) -> Path:
        return self.root / "training"

    @property
    def model_reference(self) -> Path:
        return self.training / "model_reference.json"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def score_reference(self) -> Path:
        return self.scores / "cache_reference.json"

    @property
    def decisions(self) -> Path:
        return self.root / "decisions"

    @property
    def threshold_records(self) -> Path:
        return self.decisions / "threshold_record.jsonl"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def metric_records(self) -> Path:
        return self.metrics / "metric_record.jsonl"

    @property
    def federation_metrics(self) -> Path:
        return self.metrics / "federation.json"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def verification(self) -> Path:
        return self.root / "verification"

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


class OutputsLayout:
    """Own every reserved outputs/ path name so no module hardcodes one."""

    def __init__(self, outputs_root: Path = Path("outputs")) -> None:
        self.outputs_root = outputs_root

    @property
    def runs(self) -> Path:
        return self.outputs_root / "runs"

    @property
    def cache(self) -> Path:
        return self.outputs_root / "cache"

    @property
    def cache_models(self) -> Path:
        return self.cache / "models"

    @property
    def cache_scores(self) -> Path:
        return self.cache / "scores"

    @property
    def cache_analysis(self) -> Path:
        return self.cache / "analysis"

    @property
    def campaigns(self) -> Path:
        return self.outputs_root / "campaigns"

    @property
    def logs(self) -> Path:
        return self.outputs_root / "logs"

    @property
    def monitoring(self) -> Path:
        return self.outputs_root / "monitoring"

    @property
    def reports(self) -> Path:
        return self.outputs_root / "reports"

    @property
    def environment_file(self) -> Path:
        return self.outputs_root / "environment.json"

    @property
    def telemetry_file(self) -> Path:
        return self.monitoring / "telemetry.jsonl"

    @property
    def benchmark_report(self) -> Path:
        return self.reports / "latest" / "benchmark.json"

    @property
    def readiness_plans_file(self) -> Path:
        return self.cache_analysis / "readiness_plans.json"

    @property
    def mismatch_cutoffs_file(self) -> Path:
        return self.cache_analysis / "mismatch_cutoffs.json"

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
    detector_label = (
        "ae" if detector.id is DetectorId.AUTOENCODER else detector.id.value
    )
    prefix = (
        f"{config.dataset.id.value}__{detector}__ms{int(model_seed)}__"
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
            ArtifactType.DATASET_MANIFEST: layout.data / "dataset_manifest.json",
            ArtifactType.ELIGIBILITY_MANIFEST: layout.data / "eligibility.json",
            ArtifactType.SPLIT_MANIFEST: layout.data / "calibration_assignment.json",
            ArtifactType.PREPROCESSING_MANIFEST: layout.data / "preprocessing.json",
            ArtifactType.TRAINING_MANIFEST: layout.training / "training.json",
            ArtifactType.MODEL: layout.model_reference,
            ArtifactType.SCORE_MANIFEST: layout.scores / "manifest.json",
            ArtifactType.THRESHOLD_RECORDS: layout.threshold_records,
            ArtifactType.METRICS: layout.metric_records,
            ArtifactType.VERIFICATION: layout.verification / "hashes.json",
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
        missing: list[str] = []
        mismatched: list[str] = []
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
                layout.verification / "hashes.json",
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
            "training/model_reference.json": layout.model_reference,
            "scores/cache_reference.json": layout.score_reference,
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
        patch_hash = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    import platform
    import sys

    import torch

    pin_payload = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "commit": commit,
        "patch_sha256": patch_hash,
    }
    pin_sha = hashlib.sha256(
        json.dumps(pin_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return GitEnvironment(
        git_commit=commit,
        git_clean=clean,
        git_patch_sha256=patch_hash,
        environment_pin_sha256=pin_sha,
        environment_pin_kind="python-torch-platform-commit",
    )
