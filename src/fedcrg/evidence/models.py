"""Pydantic models for run manifests, threshold/metric records, and cached-artifact hashes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict

from fedcrg.config import ExperimentConfig
from fedcrg.learning.federated import TrainingResult
from fedcrg.types import (
    Assurance,
    ByteCount,
    CalibrationAssignmentMode,
    CalibrationSeed,
    ClientId,
    DataRole,
    DatasetId,
    DecisionReason,
    DecisionState,
    DetectorId,
    DeviceName,
    ExperimentId,
    ExperimentStatus,
    FailureCode,
    FeatureName,
    Fpr,
    Fraction,
    Identifier,
    ModelSeed,
    PathString,
    PolicyId,
    PositiveCount,
    PValue,
    RunId,
    SampleCount,
    Sha256,
    Threshold,
    ThresholdSource,
    Tpr,
    Version,
)

Frozen = ConfigDict(frozen=True)


class SourceFileManifest(BaseModel):
    """Hash and size of one raw source file the dataset was prepared from."""

    model_config = Frozen

    relative_path: PurePosixPath
    sha256: Sha256
    size_bytes: ByteCount


class CalibrationAssignmentReference(BaseModel):
    """Hash of one persisted calibration-role assignment file."""

    model_config = Frozen

    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    relative_path: PurePosixPath
    sha256: Sha256


class RoleArtifactManifest(BaseModel):
    """Hash and row count of one client's prepared data file for a single role."""

    model_config = Frozen

    role: DataRole
    rows: PositiveCount
    row_id_sha256: Sha256
    relative_path: PurePosixPath
    file_sha256: Sha256


class ClientDatasetManifest(BaseModel):
    """Prepared-data artifacts for one client, grouped by role."""

    model_config = Frozen

    client_id: ClientId
    roles: tuple[RoleArtifactManifest, ...]

    def role(self, role: DataRole) -> RoleArtifactManifest:
        """Return this client's artifact for the given role."""
        for item in self.roles:
            if item.role is role:
                return item
        raise KeyError(role)


class PreparedDatasetManifest(BaseModel):
    """Full record of a dataset preparation run: source files, per-client artifacts, and calibration assignments."""

    model_config = Frozen

    dataset_id: DatasetId
    source_version: Version
    parser_version: Version
    data_spec_hash: Sha256
    feature_names: tuple[FeatureName, ...]
    clients: tuple[ClientDatasetManifest, ...]
    source_files: tuple[SourceFileManifest, ...]
    calibration_assignments: tuple[CalibrationAssignmentReference, ...]
    external_replication_supported: bool
    dataset_level_code: FailureCode | None = None
    created_at: datetime

    @property
    def deterministic_payload_sha256(self) -> Sha256:
        """Hash the manifest excluding ``created_at`` so re-running preparation on identical data reproduces the same digest."""
        payload = self.model_dump(
            mode="json",
            exclude={"created_at", "deterministic_payload_sha256"},
        )
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    @property
    def client_ids(self) -> tuple[ClientId, ...]:
        """Ids of every client covered by this manifest."""
        return tuple(client.client_id for client in self.clients)

    def client(self, client_id: ClientId) -> ClientDatasetManifest:
        """Return the dataset manifest for the given client."""
        for client in self.clients:
            if client.client_id == client_id:
                return client
        raise KeyError(client_id)


class ClientTrainingCount(BaseModel):
    """Number of rows one client trained on."""

    model_config = Frozen

    client_id: ClientId
    rows: PositiveCount


class TrainingManifest(BaseModel):
    """Record of a completed federated training run, tying the resulting model to its data and config hashes."""

    model_config = Frozen

    dataset_id: DatasetId
    detector_id: DetectorId
    model_seed: ModelSeed
    data_spec_hash: Sha256
    training_spec_hash: Sha256
    dataset_manifest_sha256: Sha256
    preprocessing_sha256: Sha256
    model_file_sha256: Sha256
    deep_svdd_center_sha256: Sha256 | None = None
    training_rows: tuple[ClientTrainingCount, ...]
    result: TrainingResult


class CalibrationRoleManifest(BaseModel):
    """Row count and row-id hash for one client's calibration-role assignment."""

    model_config = Frozen

    role: DataRole
    row_count: PositiveCount
    row_id_sha256: Sha256


class ClientCalibrationManifest(BaseModel):
    """Calibration-role assignment record for one client."""

    model_config = Frozen

    client_id: ClientId
    roles: tuple[CalibrationRoleManifest, ...]

    def role(self, role: DataRole) -> CalibrationRoleManifest:
        """Return this client's assignment record for the given role."""
        for item in self.roles:
            if item.role is role:
                return item
        raise KeyError(role)


class CalibrationAssignmentManifest(BaseModel):
    """Calibration-role assignment record for every client under one calibration seed."""

    model_config = Frozen

    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    clients: tuple[ClientCalibrationManifest, ...]

    def client(self, client_id: ClientId) -> ClientCalibrationManifest:
        """Return the calibration manifest for the given client."""
        for client in self.clients:
            if client.client_id == client_id:
                return client
        raise KeyError(client_id)


class EligibilityManifest(BaseModel):
    """Clients that met the eligibility rule for a dataset."""

    model_config = Frozen

    dataset_id: DatasetId
    eligible_clients: tuple[ClientId, ...]


class RunManifest(BaseModel):
    """Identity and status of one experiment run."""

    model_config = Frozen

    run_id: RunId
    experiment_id: ExperimentId
    policy_id: PolicyId
    config_hash: Sha256
    model_seed: ModelSeed
    calibration_seed: CalibrationSeed
    status: ExperimentStatus


class RunConfig(BaseModel):
    """Full resolved configuration and environment pin for one run."""

    model_config = Frozen

    run_id: RunId
    experiment_id: ExperimentId
    policy_id: PolicyId
    parameters: ExperimentConfig
    model_seed: ModelSeed
    calibration_seed: CalibrationSeed
    config_hash: Sha256
    data_spec_hash: Sha256
    training_spec_hash: Sha256
    git_commit: Identifier
    git_clean: bool
    git_patch_sha256: Sha256 | None = None
    environment_pin_sha256: Sha256
    environment_pin_kind: Identifier


class ThresholdRecord(BaseModel):
    """Per-client threshold decision produced by one policy in one run."""

    model_config = Frozen

    run_id: RunId
    policy_id: PolicyId
    client_id: ClientId
    tau_ref: Threshold
    tau_local: Threshold | None
    selected_tau: Threshold | None
    readiness_n: SampleCount
    readiness_rank: PositiveCount
    readiness_probability: Assurance
    mismatch_n: SampleCount
    mismatch_x: PositiveCount
    cp_lower: Fpr
    cp_upper: Fpr
    p_low: PValue | None
    p_high: PValue
    state: DecisionState
    tie_count: PositiveCount
    selected_source: ThresholdSource
    reason_code: DecisionReason


class MetricRecord(BaseModel):
    """Per-client detection metrics produced by one policy in one run."""

    model_config = Frozen

    run_id: RunId
    policy_id: PolicyId
    client_id: ClientId
    benign_n: PositiveCount
    attack_n: PositiveCount
    fp: PositiveCount
    tn: PositiveCount
    tp: PositiveCount
    fn: PositiveCount
    fpr: Fpr
    tpr: Tpr | None
    precision: Fpr | None
    f1: Fpr | None
    balanced_accuracy: Fpr | None
    auroc: Fpr
    auprc: Fpr
    band_error: Fraction
    attack_balanced_tpr: Tpr | None


class ChecksumRecord(BaseModel):
    """Hash of one file, identified by its path relative to the run directory."""

    model_config = Frozen

    relative_path: PathString
    sha256: Sha256


class CacheReference(BaseModel):
    """Content hash of a frozen cache artifact (e.g. a trained model or score cache), recorded so later reads can detect drift."""

    model_config = Frozen

    relative_path: Identifier
    sha256: Sha256
    size_bytes: ByteCount


class GitEnvironment(BaseModel):
    """Git commit and working-tree state captured at run time, plus a pointer to the environment pin."""

    model_config = Frozen

    git_commit: Identifier
    git_clean: bool
    git_patch_sha256: Sha256 | None = None
    environment_pin_sha256: Sha256
    environment_pin_kind: Identifier


class EnvironmentPin(BaseModel):
    """Interpreter, library, and platform versions captured at run time."""

    model_config = Frozen

    python: Version
    torch: Version
    platform: DeviceName
    commit: Identifier
    patch_sha256: Sha256 | None = None

    @property
    def sha256(self) -> Sha256:
        """Identify this exact interpreter/library/platform/commit combination for reproducibility checks."""
        serialized = self.model_dump_json().encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
