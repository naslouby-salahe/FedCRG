"""Typed persistence contract for frozen detector training evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.serialization import (
    as_json_dict,
    as_json_float,
    as_json_int,
    as_json_list,
    atomic_write_json,
)
from fedcrg.domain.enums import DatasetId, DetectorId
from fedcrg.domain.identifiers import ClientId, ModelSeed, Sha256
from fedcrg.federated.models import ClientRoundResult, RoundResult, TrainingResult


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
