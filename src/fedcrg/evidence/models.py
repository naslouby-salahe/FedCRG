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
    model_config = Frozen

    relative_path: PurePosixPath
    sha256: Sha256
    size_bytes: ByteCount


class CalibrationAssignmentReference(BaseModel):
    model_config = Frozen

    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    relative_path: PurePosixPath
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

    @property
    def deterministic_payload_sha256(self) -> Sha256:
        payload = self.model_dump(
            mode="json",
            exclude={"created_at", "deterministic_payload_sha256"},
        )
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

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


class RunConfig(BaseModel):
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


class ChecksumRecord(BaseModel):
    model_config = Frozen

    relative_path: PathString
    sha256: Sha256


class EmpiricalPolicyResult(BaseModel):
    model_config = Frozen

    run_id: RunId
    policy_id: PolicyId
    model_seed: ModelSeed
    calibration_seed: CalibrationSeed
    config_hash: Sha256
    mebe: Fraction
    high_excess: Fraction
    band_violation_rate: Fraction
    attack_balanced_macro_tpr: Tpr | None


class EmpiricalExperimentResult(BaseModel):
    model_config = Frozen

    experiment_id: ExperimentId
    config_hash: Sha256
    records: tuple[EmpiricalPolicyResult, ...]


class ExperimentProvenance(BaseModel):
    model_config = Frozen

    experiment_id: ExperimentId
    config_hash: Sha256
    data_spec_hash: Sha256
    source_digests: tuple[ChecksumRecord, ...]


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


class EnvironmentPin(BaseModel):
    model_config = Frozen

    python: Version
    torch: Version
    platform: DeviceName
    commit: Identifier
    patch_sha256: Sha256 | None = None

    @property
    def sha256(self) -> Sha256:
        serialized = self.model_dump_json().encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
