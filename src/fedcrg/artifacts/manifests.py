"""Typed persistence for run, dataset, eligibility, preprocessing, and training manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from fedcrg.artifacts.json_io import (
    as_json_dict,
    as_json_float,
    as_json_int,
    as_json_list,
    atomic_write_json,
    to_json_value,
)
from fedcrg.datasets.eligibility import EligibilityManifest, EligibilityRecord
from fedcrg.datasets.prepare import (
    CalibrationAssignmentReference,
    ClientDatasetManifest,
    PreparedDatasetManifest,
    RoleArtifactManifest,
    SourceFileManifest,
)
from fedcrg.datasets.preprocessing import ClientPreprocessingParameters, PreprocessingModel
from fedcrg.datasets.splits import (
    CalibrationAssignmentManifest,
    CalibrationRoleManifest,
    ClientCalibrationManifest,
)
from fedcrg.domain.enums import (
    CalibrationAssignmentMode,
    ChronologyStatus,
    DataRole,
    DatasetId,
    DetectorId,
    EligibilityStatus,
    ExperimentId,
    ExperimentStatus,
    FailureCode,
    PolicyId,
)
from fedcrg.domain.errors import ImmutableRunError
from fedcrg.domain.identifiers import CalibrationSeed, ClientId, ModelSeed, RunId, Sha256
from fedcrg.federation.training_results import ClientRoundResult, RoundResult, TrainingResult


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: RunId
    experiment_id: ExperimentId
    policy_id: PolicyId
    config_hash: Sha256
    model_seed: ModelSeed
    calibration_seed: CalibrationSeed
    status: ExperimentStatus


class RunManifestStore:
    def load(self, path: Path) -> RunManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return RunManifest(
            run_id=RunId(str(raw["run_id"])),
            experiment_id=ExperimentId(str(raw["experiment_id"])),
            policy_id=PolicyId(str(raw["policy_id"])),
            config_hash=Sha256(str(raw["config_hash"])),
            model_seed=ModelSeed(int(raw["model_seed"])),
            calibration_seed=CalibrationSeed(int(raw["calibration_seed"])),
            status=ExperimentStatus(str(raw["status"])),
        )

    def save(self, path: Path, manifest: RunManifest) -> None:
        if path.exists() and self.load(path).status is ExperimentStatus.COMPLETE:
            raise ImmutableRunError(f"Completed run is immutable: {path.parent}")
        atomic_write_json(path, manifest)


class PreparedDatasetManifestStore:
    def build(
        self,
        *,
        dataset_id: DatasetId,
        source_version: str,
        parser_version: str,
        data_spec_hash: Sha256,
        feature_names: tuple[str, ...],
        clients: tuple[ClientDatasetManifest, ...],
        source_files: tuple[SourceFileManifest, ...],
        calibration_assignments: tuple[CalibrationAssignmentReference, ...],
        external_replication_supported: bool,
        dataset_level_code: FailureCode | None,
    ) -> PreparedDatasetManifest:
        deterministic = {
            "dataset_id": dataset_id,
            "source_version": source_version,
            "parser_version": parser_version,
            "data_spec_hash": data_spec_hash,
            "feature_names": feature_names,
            "clients": clients,
            "source_files": source_files,
            "calibration_assignments": calibration_assignments,
            "external_replication_supported": external_replication_supported,
            "dataset_level_code": dataset_level_code,
        }
        serialized = json.dumps(
            to_json_value(deterministic),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return PreparedDatasetManifest(
            dataset_id=dataset_id,
            source_version=source_version,
            parser_version=parser_version,
            data_spec_hash=data_spec_hash,
            feature_names=feature_names,
            clients=clients,
            source_files=source_files,
            calibration_assignments=calibration_assignments,
            external_replication_supported=external_replication_supported,
            dataset_level_code=dataset_level_code,
            created_at=datetime.now(UTC),
            deterministic_payload_sha256=Sha256(hashlib.sha256(serialized).hexdigest()),
        )

    def save(self, path: Path, manifest: PreparedDatasetManifest) -> None:
        atomic_write_json(path, manifest)

    def load(self, path: Path) -> PreparedDatasetManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        clients = tuple(
            ClientDatasetManifest(
                client_id=ClientId(str(client["client_id"])),
                roles=tuple(
                    RoleArtifactManifest(
                        role=DataRole(str(role["role"])),
                        rows=int(role["rows"]),
                        row_id_sha256=Sha256(str(role["row_id_sha256"])),
                        relative_path=PurePosixPath(str(role["relative_path"])),
                        file_sha256=Sha256(str(role["file_sha256"])),
                    )
                    for role in client["roles"]
                ),
            )
            for client in raw["clients"]
        )
        sources = tuple(
            SourceFileManifest(
                relative_path=PurePosixPath(str(item["relative_path"])),
                sha256=Sha256(str(item["sha256"])),
                size_bytes=int(item["size_bytes"]),
            )
            for item in raw["source_files"]
        )
        assignments = tuple(
            CalibrationAssignmentReference(
                calibration_seed=CalibrationSeed(int(item["calibration_seed"])),
                mode=CalibrationAssignmentMode(str(item["mode"])),
                sha256=Sha256(str(item["sha256"])),
            )
            for item in raw["calibration_assignments"]
        )
        code = raw["dataset_level_code"]
        manifest = PreparedDatasetManifest(
            dataset_id=DatasetId(str(raw["dataset_id"])),
            source_version=str(raw["source_version"]),
            parser_version=str(raw["parser_version"]),
            data_spec_hash=Sha256(str(raw["data_spec_hash"])),
            feature_names=tuple(str(value) for value in raw["feature_names"]),
            clients=clients,
            source_files=sources,
            calibration_assignments=assignments,
            external_replication_supported=bool(raw["external_replication_supported"]),
            dataset_level_code=None if code is None else FailureCode(str(code)),
            created_at=datetime.fromisoformat(str(raw["created_at"])),
            deterministic_payload_sha256=Sha256(str(raw["deterministic_payload_sha256"])),
        )
        rebuilt = self.build(
            dataset_id=manifest.dataset_id,
            source_version=manifest.source_version,
            parser_version=manifest.parser_version,
            data_spec_hash=manifest.data_spec_hash,
            feature_names=manifest.feature_names,
            clients=manifest.clients,
            source_files=manifest.source_files,
            calibration_assignments=manifest.calibration_assignments,
            external_replication_supported=manifest.external_replication_supported,
            dataset_level_code=manifest.dataset_level_code,
        )
        if rebuilt.deterministic_payload_sha256 != manifest.deterministic_payload_sha256:
            raise ValueError("Prepared dataset deterministic payload hash mismatch")
        return manifest


class EligibilityManifestStore:
    def save(self, path: Path, manifest: EligibilityManifest) -> None:
        atomic_write_json(path, manifest)

    def load(self, path: Path) -> EligibilityManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return EligibilityManifest(
            dataset_id=DatasetId(str(raw["dataset_id"])),
            discovered_clients=tuple(ClientId(str(value)) for value in raw["discovered_clients"]),
            eligible_clients=tuple(ClientId(str(value)) for value in raw["eligible_clients"]),
            records=tuple(
                EligibilityRecord(
                    client_id=ClientId(str(item["client_id"])),
                    status=EligibilityStatus(str(item["status"])),
                    benign_count=int(item["benign_count"]),
                    malicious_count=int(item["malicious_count"]),
                    attack_development_capacity=int(item["attack_development_capacity"]),
                    primary_code=(
                        None
                        if item["primary_code"] is None
                        else FailureCode(str(item["primary_code"]))
                    ),
                    secondary_codes=tuple(
                        FailureCode(str(code)) for code in item["secondary_codes"]
                    ),
                    chronology=ChronologyStatus(str(item["chronology"])),
                )
                for item in raw["records"]
            ),
        )


class CalibrationAssignmentManifestStore:
    def save(self, path: Path, manifest: CalibrationAssignmentManifest) -> None:
        atomic_write_json(path, manifest)

    def load(self, path: Path) -> CalibrationAssignmentManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return CalibrationAssignmentManifest(
            calibration_seed=CalibrationSeed(int(raw["calibration_seed"])),
            mode=CalibrationAssignmentMode(str(raw["mode"])),
            clients=tuple(
                ClientCalibrationManifest(
                    client_id=ClientId(str(client["client_id"])),
                    roles=tuple(
                        CalibrationRoleManifest(
                            role=DataRole(str(role["role"])),
                            row_count=int(role["row_count"]),
                            row_id_sha256=Sha256(str(role["row_id_sha256"])),
                        )
                        for role in client["roles"]
                    ),
                )
                for client in raw["clients"]
            ),
        )


class PreprocessingManifestStore:
    """Persist and validate the exact train-only preprocessing contract."""

    def save(self, path: Path, model: PreprocessingModel) -> None:
        atomic_write_json(path, model)

    def load(self, path: Path) -> PreprocessingModel:
        raw = json.loads(path.read_text(encoding="utf-8"))
        clients = tuple(
            ClientPreprocessingParameters(
                client_id=ClientId(str(item["client_id"])),
                training_row_sha256=Sha256(str(item["training_row_sha256"])),
                medians=(
                    None
                    if item["medians"] is None
                    else tuple(float(value) for value in item["medians"])
                ),
            )
            for item in raw["clients"]
        )
        model = PreprocessingModel(
            dataset=DatasetId(str(raw["dataset"])),
            feature_columns=tuple(str(value) for value in raw["feature_columns"]),
            clients=clients,
            global_minima=tuple(float(value) for value in raw["global_minima"]),
            global_maxima=tuple(float(value) for value in raw["global_maxima"]),
        )
        expected_constant = tuple(bool(value) for value in raw["constant_features"])
        if model.constant_features != expected_constant:
            raise ValueError("Preprocessing constant-feature ledger mismatch")
        if int(raw["extrema_upload_bytes_per_client"]) != model.extrema_upload_bytes_per_client:
            raise ValueError("Preprocessing per-client communication ledger mismatch")
        if int(raw["extrema_upload_bytes_total"]) != model.extrema_upload_bytes_total:
            raise ValueError("Preprocessing total communication ledger mismatch")
        return model


@dataclass(frozen=True, slots=True)
class ClientTrainingCount:
    client_id: ClientId
    rows: int


@dataclass(frozen=True, slots=True)
class TrainingManifest:
    dataset_id: DatasetId
    detector_id: DetectorId
    model_seed: ModelSeed
    data_spec_hash: Sha256
    training_spec_hash: Sha256
    dataset_manifest_sha256: Sha256
    preprocessing_sha256: Sha256
    model_file_sha256: Sha256
    deep_svdd_center_sha256: Sha256 | None
    training_rows: tuple[ClientTrainingCount, ...]
    result: TrainingResult


class TrainingManifestStore:
    def save(self, path: Path, manifest: TrainingManifest) -> None:
        atomic_write_json(path, manifest)

    def load(self, path: Path) -> TrainingManifest:
        raw = as_json_dict(json.loads(path.read_text(encoding="utf-8")))
        result_raw = as_json_dict(raw["result"])
        rounds = tuple(
            self._round(as_json_dict(item)) for item in as_json_list(result_raw["rounds"])
        )
        correlation_raw = result_raw["round20_training_score_correlation"]
        result = TrainingResult(
            model_seed=ModelSeed(as_json_int(result_raw["model_seed"])),
            rounds=rounds,
            final_model_hash=Sha256(str(result_raw["final_model_hash"])),
            trainable_parameter_count=as_json_int(result_raw["trainable_parameter_count"]),
            model_payload_bytes=as_json_int(result_raw["model_payload_bytes"]),
            total_model_communication_bytes=as_json_int(
                result_raw["total_model_communication_bytes"]
            ),
            round20_training_score_correlation=(
                None if correlation_raw is None else as_json_float(correlation_raw)
            ),
        )
        center_hash_raw = raw["deep_svdd_center_sha256"]
        manifest = TrainingManifest(
            dataset_id=DatasetId(str(raw["dataset_id"])),
            detector_id=DetectorId(str(raw["detector_id"])),
            model_seed=ModelSeed(as_json_int(raw["model_seed"])),
            data_spec_hash=Sha256(str(raw["data_spec_hash"])),
            training_spec_hash=Sha256(str(raw["training_spec_hash"])),
            dataset_manifest_sha256=Sha256(str(raw["dataset_manifest_sha256"])),
            preprocessing_sha256=Sha256(str(raw["preprocessing_sha256"])),
            model_file_sha256=Sha256(str(raw["model_file_sha256"])),
            deep_svdd_center_sha256=(
                None if center_hash_raw is None else Sha256(str(center_hash_raw))
            ),
            training_rows=tuple(
                ClientTrainingCount(
                    ClientId(str(as_json_dict(item)["client_id"])),
                    as_json_int(as_json_dict(item)["rows"]),
                )
                for item in as_json_list(raw["training_rows"])
            ),
            result=result,
        )
        if manifest.model_seed != manifest.result.model_seed:
            raise ValueError("Training manifest/result model seed mismatch")
        return manifest

    @staticmethod
    def _round(raw: dict[str, object]) -> RoundResult:
        clients = tuple(
            ClientRoundResult(
                client_id=ClientId(str(as_json_dict(item)["client_id"])),
                mean_loss=as_json_float(as_json_dict(item)["mean_loss"]),
                record_presentations=as_json_int(as_json_dict(item)["record_presentations"]),
                optimizer_steps=as_json_int(as_json_dict(item)["optimizer_steps"]),
                model_hash=Sha256(str(as_json_dict(item)["model_hash"])),
            )
            for item in as_json_list(raw["client_results"])
        )
        return RoundResult(
            round_index=as_json_int(raw["round_index"]),
            learning_rate=as_json_float(raw["learning_rate"]),
            selected_clients=tuple(
                ClientId(str(value)) for value in as_json_list(raw["selected_clients"])
            ),
            client_results=clients,
            mean_client_loss=as_json_float(raw["mean_client_loss"]),
            minimum_client_loss=as_json_float(raw["minimum_client_loss"]),
            maximum_client_loss=as_json_float(raw["maximum_client_loss"]),
            parameter_update_norm=as_json_float(raw["parameter_update_norm"]),
            model_payload_bytes=as_json_int(raw["model_payload_bytes"]),
            round_communication_bytes=as_json_int(raw["round_communication_bytes"]),
            global_model_hash=Sha256(str(raw["global_model_hash"])),
        )
