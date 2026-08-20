"""File-backed persistence for evidence manifests, plus atomic writes and artifact verification."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

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
from fedcrg.paths import RunLayout
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


def _jsonable(value: object) -> JsonValue:
    """Recursively coerce a value to JSON-safe types, falling back to ``str()`` for anything without a dedicated case (e.g. enums, paths)."""
    match value:
        case BaseModel():
            return value.model_dump(mode="json")
        case tuple() as items:
            return [_jsonable(item) for item in items]
        case list() as items:
            return [_jsonable(item) for item in items]
        case dict():
            return {str(key): _jsonable(item) for key, item in value.items()}
        case str() | int() | float() | bool() | None:
            return value
        case _:
            return str(value)


@contextlib.contextmanager
def atomic_file(path: Path):
    """Write through a sibling temp file and ``os.replace`` so readers never observe a partially written artifact; the temp file is removed if writing fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            yield handle
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_json(path: Path, payload: object) -> None:
    """Write `payload` as pretty-printed, sorted-key JSON atomically."""
    with atomic_file(path) as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def atomic_write_text(path: Path, content: str) -> None:
    """Write `content` to `path` atomically."""
    with atomic_file(path) as handle:
        handle.write(content)


def write_jsonl(path: Path, records: tuple[BaseModel, ...]) -> None:
    """Write one JSON object per line, atomically."""
    with atomic_file(path) as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")


def load_json_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    """Parse a JSON file into the given pydantic model type."""
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def load_yaml_mapping(path: Path) -> object:
    """Load a YAML file and require its top-level document to be a mapping."""
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration document must be a mapping: {path}")
    return {str(key): value for key, value in raw.items()}


class ModelStore[ModelT: BaseModel]:
    """Atomic JSON persistence for a single pydantic model type."""

    model: type[ModelT]

    def save(self, path: Path, manifest: ModelT) -> None:
        """Write the manifest to `path` atomically."""
        atomic_write_json(path, manifest)

    def load(self, path: Path) -> ModelT:
        """Read and parse the manifest at `path`."""
        return load_json_model(path, self.model)

    def load_model(self, path: Path) -> ModelT:
        """Read and parse the manifest at `path`."""
        return load_json_model(path, self.model)


class PreparedDatasetManifestStore(ModelStore[PreparedDatasetManifest]):
    """Persistence for `PreparedDatasetManifest`."""

    model = PreparedDatasetManifest


class TrainingManifestStore(ModelStore[TrainingManifest]):
    """Persistence for `TrainingManifest`."""

    model = TrainingManifest


class RunManifestStore(ModelStore[RunManifest]):
    """Persistence for `RunManifest`, refusing to silently overwrite a finalized run."""

    model = RunManifest
    _TERMINAL_STATUSES = frozenset({ExperimentStatus.COMPLETE, ExperimentStatus.FAILED})

    def save(self, path: Path, manifest: RunManifest) -> None:
        """Refuse to overwrite a run manifest that already reached a terminal status with a different status."""
        if path.is_file():
            try:
                existing = self.load(path)
                if (
                    existing.status in self._TERMINAL_STATUSES
                    and existing.status is not manifest.status
                ):
                    from fedcrg.types import ImmutableRunError

                    raise ImmutableRunError(
                        f"Cannot overwrite a finalized run manifest: {path} "
                        f"({existing.status} -> {manifest.status})"
                    )
            except (FileNotFoundError, ValueError):
                pass
        super().save(path, manifest)


class EligibilityManifestStore(ModelStore[EligibilityManifest]):
    """Persistence for `EligibilityManifest`."""

    model = EligibilityManifest


class CalibrationAssignmentManifestStore(ModelStore[CalibrationAssignmentManifest]):
    """Persistence for `CalibrationAssignmentManifest`."""

    model = CalibrationAssignmentManifest


class CacheReferenceStore(ModelStore[CacheReference]):
    """Persistence for `CacheReference`."""

    model = CacheReference

    @staticmethod
    def build(path: Path, outputs_root: Path) -> CacheReference:
        """Hash a file and record its path relative to the outputs root."""
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


def build_run_id(
    config: ExperimentConfig,
    model_seed: ModelSeed,
    calibration_seed: CalibrationSeed,
    policy: PolicyId,
) -> RunId:
    """Encode the run's identity into a stable, filesystem-safe string; float parameters are rounded to fixed-precision integers so the id is reproducible."""
    protocol = config.protocol
    if config.detector is None:
        raise ValueError("Run identity requires a real-data detector profile")

    alpha_ppm = round(protocol.alpha * 1_000_000)
    rho_bp = round(protocol.rho * 10_000)
    assurance_bp = round(protocol.readiness_assurance * 10_000)
    confidence_bp = round(protocol.mismatch_confidence * 10_000)
    detector_label = "ae" if config.detector.id is DetectorId.AUTOENCODER else config.detector.id

    return (
        f"{config.dataset.id}__{detector_label}__ms{int(model_seed)}__"
        f"cs{int(calibration_seed)}__a{alpha_ppm}__r{rho_bp}__"
        f"ga{assurance_bp}__gb{confidence_bp}__{policy.lower()}__"
        f"cfg{config.config_hash[:12]}"
    )


@dataclass(frozen=True, slots=True)
class FileHashRecord:
    """Hash of one run artifact, identified by its path relative to the run root."""

    relative_path: Identifier
    sha256: Sha256


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of checking a run's required artifacts for presence and hash consistency."""

    valid: bool
    missing: tuple[Identifier, ...]
    mismatched: tuple[Identifier, ...]
    hashes: tuple[FileHashRecord, ...]

    def hash_for(self, relative_path: Identifier) -> Sha256 | None:
        """Return the recorded hash for a file, or None if it wasn't checked."""
        return next(
            (record.sha256 for record in self.hashes if record.relative_path == relative_path), None
        )


class ArtifactVerifier:
    """Checks that a run's required artifacts exist and match their recorded hashes."""

    def _path_for(self, layout: RunLayout, artifact: ArtifactType) -> Path | None:
        match artifact:
            case ArtifactType.RESOLVED_CONFIG:
                return layout.resolved_config
            case ArtifactType.DATASET_MANIFEST:
                return layout.dataset_manifest
            case ArtifactType.ELIGIBILITY_MANIFEST:
                return layout.eligibility_manifest
            case ArtifactType.SPLIT_MANIFEST:
                return layout.split_manifest
            case ArtifactType.PREPROCESSING_MANIFEST:
                return layout.preprocessing_evidence
            case ArtifactType.TRAINING_MANIFEST:
                return layout.training_manifest
            case ArtifactType.MODEL:
                return layout.model_reference
            case ArtifactType.SCORE_MANIFEST:
                return layout.score_manifest
            case ArtifactType.THRESHOLD_RECORDS:
                return layout.threshold_records
            case ArtifactType.METRICS:
                return layout.metric_records
            case ArtifactType.VERIFICATION:
                return layout.hashes
            case _:
                return None

    def required_files(self, layout: RunLayout, definition: ExperimentSpec) -> tuple[Path, ...]:
        """List the artifact paths a run of this kind must produce."""
        from fedcrg.evidence.contracts import experiment_contract

        required = [
            layout.run_config,
            layout.resolved_config,
            layout.environment,
            layout.manifest,
        ]
        contract = experiment_contract(definition.id)
        artifacts = contract.run_artifacts or definition.required_evidence
        for artifact in artifacts:
            if path := self._path_for(layout, artifact):
                required.append(path)
        return tuple(required)

    def record(self, layout: RunLayout, definition: ExperimentSpec) -> VerificationResult:
        """Hash every required artifact, compare against recorded references, and write the verification file."""
        missing: list[PathString] = []
        mismatched: list[PathString] = []
        hashes: list[FileHashRecord] = []

        for path in self.required_files(layout, definition):
            relative = path.relative_to(layout.root).as_posix()
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(relative)
                continue

            digest = sha256_file(path)
            hashes.append(FileHashRecord(relative, digest))

        for pointer in (layout.model_reference, layout.score_reference):
            mismatch = self._cache_pointer_mismatch(layout, pointer)
            if mismatch is not None:
                mismatched.append(mismatch)

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
    def _cache_pointer_mismatch(layout: RunLayout, pointer: Path) -> PathString | None:
        """Compare a cache pointer to its target file. Unreadable pointers are ignored."""
        if not pointer.is_file():
            return None
        try:
            reference = CacheReference.model_validate_json(pointer.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError):
            return None
        outputs_root = layout.root.parent.parent
        target = Path(reference.relative_path)
        if not target.is_absolute():
            target = outputs_root / reference.relative_path
        relative = pointer.relative_to(layout.root).as_posix()
        if not target.is_file() or sha256_file(target) != reference.sha256:
            return relative
        return None


def capture_environment(repository_root: Path) -> GitEnvironment:
    """Snapshot the current git commit/dirty state and interpreter/library versions."""
    import torch

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=str(repository_root),
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    commit = run("rev-parse", "HEAD")
    clean = not run("status", "--porcelain")
    patch = run("diff")
    patch_hash = sha256_text(patch) if patch else None

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
