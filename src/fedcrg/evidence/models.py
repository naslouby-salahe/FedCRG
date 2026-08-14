"""Persisted evidence schemas: run manifests, threshold/metric records,
training manifests, prepared-dataset manifests, and environment pins."""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from fedcrg.learning.federated import TrainingResult
from fedcrg.learning.scores import ScoreManifest
from fedcrg.types import (
    Alpha,
    Assurance,
    ByteCount,
    CalibrationAssignmentMode,
    CalibrationReadinessState,
    CalibrationSeed,
    ClientId,
    ConfidenceInterval,
    DataRole,
    DatasetId,
    DecisionReason,
    DecisionState,
    DetectorId,
    ExperimentId,
    ExperimentStatus,
    FailureCode,
    FeatureName,
    Fpr,
    Fraction,
    Identifier,
    MismatchOutcome,
    ModelSeed,
    PValue,
    PolicyId,
    PositiveCount,
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
    model_config = Frozen

    relative_path: PurePosixPath
    sha256: Sha256
    size_bytes: ByteCount


class CalibrationAssignmentReference(BaseModel):
    model_config = Frozen

    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    sha256: Sha256


class RoleArtifactManifest(BaseModel):
    model_config = Frozen

    role: DataRole
    rows: PositiveCount
    row_id_sha256: Sha256
    relative_path: PurePosixPath
    file_sha256: Sha256


class ClientDatasetManifest(BaseModel):
    model_config = Frozen

    client_id: ClientId
    roles: tuple[RoleArtifactManifest, ...]

    def role(self, role: DataRole) -> RoleArtifactManifest:
        for item in self.roles:
            if item.role is role:
                return item
        raise KeyError(role)


class PreparedDatasetManifest(BaseModel):
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
    deterministic_payload_sha256: Sha256

    @property
    def client_ids(self) -> tuple[ClientId, ...]:
        return tuple(client.client_id for client in self.clients)

    def client(self, client_id: ClientId) -> ClientDatasetManifest:
        for client in self.clients:
            if client.client_id == client_id:
                return client
        raise KeyError(client_id)


class ClientTrainingCount(BaseModel):
    model_config = Frozen

    client_id: ClientId
    rows: PositiveCount


class TrainingManifest(BaseModel):
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
    model_config = Frozen

    role: DataRole
    row_count: PositiveCount
    row_id_sha256: Sha256


class ClientCalibrationManifest(BaseModel):
    model_config = Frozen

    client_id: ClientId
    roles: tuple[CalibrationRoleManifest, ...]

    def role(self, role: DataRole) -> CalibrationRoleManifest:
        for item in self.roles:
            if item.role is role:
                return item
        raise KeyError(role)


class CalibrationAssignmentManifest(BaseModel):
    model_config = Frozen

    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    clients: tuple[ClientCalibrationManifest, ...]

    def client(self, client_id: ClientId) -> ClientCalibrationManifest:
        for client in self.clients:
            if client.client_id == client_id:
                return client
        raise KeyError(client_id)


class EligibilityManifest(BaseModel):
    model_config = Frozen

    dataset_id: DatasetId
    eligible_clients: tuple[ClientId, ...]


class RunManifest(BaseModel):
    model_config = Frozen

    run_id: RunId
    experiment_id: ExperimentId
    policy_id: PolicyId
    config_hash: Sha256
    model_seed: ModelSeed
    calibration_seed: CalibrationSeed
    status: ExperimentStatus


class ThresholdRecord(BaseModel):
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


class CacheReference(BaseModel):
    model_config = Frozen

    relative_path: Identifier
    sha256: Sha256
    size_bytes: ByteCount


class GitEnvironment(BaseModel):
    model_config = Frozen

    git_commit: Identifier
    git_clean: bool
    git_patch_sha256: Sha256 | None = None
    environment_pin_sha256: Sha256
    environment_pin_kind: Identifier
