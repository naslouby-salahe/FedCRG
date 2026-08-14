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

ModelT = TypeVar("ModelT", bound=BaseModel)


def _jsonable(value: object) -> JsonValue:
    match value:
        case BaseModel():
            return value.model_dump(mode="json")
        case tuple() | list():
            return [_jsonable(item) for item in value]
        case dict():
            return {str(key): _jsonable(item) for key, item in value.items()}
        case str() | int() | float() | bool() | None:
            return value
        case _:
            return str(value)


@contextlib.contextmanager
def atomic_file(path: Path):
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
    with atomic_file(path) as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def atomic_write_text(path: Path, content: str) -> None:
    with atomic_file(path) as handle:
        handle.write(content)


def write_jsonl(path: Path, records: tuple[BaseModel, ...]) -> None:
    with atomic_file(path) as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")


def load_json_model(path: Path, model: type[ModelT]) -> ModelT:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def load_yaml_mapping(path: Path) -> object:
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration document must be a mapping: {path}")
    return {str(key): value for key, value in raw.items()}


class ModelStore(Generic[ModelT]):
    model: type[ModelT]

    def save(self, path: Path, manifest: ModelT) -> None:
        atomic_write_json(path, manifest)

    def load(self, path: Path) -> ModelT:
        return load_json_model(path, self.model)

    def load_model(self, path: Path) -> ModelT:
        return load_json_model(path, self.model)


class PreparedDatasetManifestStore(ModelStore[PreparedDatasetManifest]):
    model = PreparedDatasetManifest


class TrainingManifestStore(ModelStore[TrainingManifest]):
    model = TrainingManifest


class RunManifestStore(ModelStore[RunManifest]):
    model = RunManifest
    _TERMINAL_STATUSES = frozenset({ExperimentStatus.COMPLETE.value, ExperimentStatus.FAILED.value})

    def save(self, path: Path, manifest: RunManifest) -> None:
        if path.is_file():
            try:
                existing = self.load(path)
                if (
                    existing.status.value in self._TERMINAL_STATUSES
                    and existing.status is not manifest.status
                ):
                    from fedcrg.types import ImmutableRunError
                    raise ImmutableRunError(
                        f"Cannot overwrite a finalized run manifest: {path} "
                        f"({existing.status.value} -> {manifest.status.value})"
                    )
            except (FileNotFoundError, ValueError):
                pass
        super().save(path, manifest)


class EligibilityManifestStore(ModelStore[EligibilityManifest]):
    model = EligibilityManifest


class CalibrationAssignmentManifestStore(ModelStore[CalibrationAssignmentManifest]):
    model = CalibrationAssignmentManifest


class CacheReferenceStore(ModelStore[CacheReference]):
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


def build_run_id(
    config: ExperimentConfig,
    model_seed: ModelSeed,
    calibration_seed: CalibrationSeed,
    policy: PolicyId,
) -> RunId:
    protocol = config.protocol
    if config.detector is None:
        raise ValueError("Run identity requires a real-data detector profile")

    alpha_ppm = round(protocol.alpha * 1_000_000)
    rho_bp = round(protocol.rho * 10_000)
    assurance_bp = round(protocol.readiness_assurance * 10_000)
    confidence_bp = round(protocol.mismatch_confidence * 10_000)
    detector_label = "ae" if config.detector.id is DetectorId.AUTOENCODER else config.detector.id.value

    return (
        f"{config.dataset.id.value}__{detector_label}__ms{int(model_seed)}__"
        f"cs{int(calibration_seed)}__a{alpha_ppm}__r{rho_bp}__"
        f"ga{assurance_bp}__gb{confidence_bp}__{policy.value.lower()}__"
        f"cfg{config.config_hash[:12]}"
    )


@dataclass(frozen=True, slots=True)
class FileHashRecord:
    relative_path: Identifier
    sha256: Sha256


@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool
    missing: tuple[Identifier, ...]
    mismatched: tuple[Identifier, ...]
    hashes: tuple[FileHashRecord, ...]

    def hash_for(self, relative_path: Identifier) -> Sha256 | None:
        return next((record.sha256 for record in self.hashes if record.relative_path == relative_path), None)


class ArtifactVerifier:
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
        required = [
            layout.run_config,
            layout.resolved_config,
            layout.environment,
            layout.manifest,
        ]
        for artifact in definition.required_evidence:
            if path := self._path_for(layout, artifact):
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

            if (expected := self._expected_hash(layout, relative)) and digest != expected:
                mismatched.append(relative)

        if layout.verification.is_dir():
            atomic_write_json(
                layout.hashes,
                {"files": [{"relative_path": item.relative_path, "sha256": item.sha256} for item in hashes]},
            )

        return VerificationResult(
            valid=not missing and not mismatched,
            missing=tuple(missing),
            mismatched=tuple(mismatched),
            hashes=tuple(hashes),
        )

    @staticmethod
    def _expected_hash(layout: RunLayout, relative: Identifier) -> Sha256 | None:
        reference = None
        if relative == layout.model_reference.relative_to(layout.root).as_posix():
            reference = layout.model_reference
        elif relative == layout.score_reference.relative_to(layout.root).as_posix():
            reference = layout.score_reference

        if reference is None or not reference.is_file():
            return None

        try:
            return CacheReference.model_validate_json(reference.read_text(encoding="utf-8")).sha256
        except (ValueError, json.JSONDecodeError):
            return None


def capture_environment(repository_root: Path) -> GitEnvironment:
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
