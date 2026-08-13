"""Typed persistence contract for frozen detector training evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.core.enums import DatasetId, DetectorId
from fedcrg.core.ids import ClientId, ModelSeed, Sha256
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
        raw = json.loads(path.read_text(encoding="utf-8"))
        result_raw = raw["result"]
        rounds = tuple(self._round(item) for item in result_raw["rounds"])
        result = TrainingResult(
            model_seed=ModelSeed(int(result_raw["model_seed"])),
            rounds=rounds,
            final_model_hash=Sha256(str(result_raw["final_model_hash"])),
            trainable_parameter_count=int(result_raw["trainable_parameter_count"]),
            model_payload_bytes=int(result_raw["model_payload_bytes"]),
            total_model_communication_bytes=int(
                result_raw["total_model_communication_bytes"]
            ),
            round20_training_score_correlation=(
                None
                if result_raw["round20_training_score_correlation"] is None
                else float(result_raw["round20_training_score_correlation"])
            ),
        )
        manifest = TrainingManifest(
            dataset_id=DatasetId(str(raw["dataset_id"])),
            detector_id=DetectorId(str(raw["detector_id"])),
            model_seed=ModelSeed(int(raw["model_seed"])),
            data_spec_hash=Sha256(str(raw["data_spec_hash"])),
            training_spec_hash=Sha256(str(raw["training_spec_hash"])),
            dataset_manifest_sha256=Sha256(str(raw["dataset_manifest_sha256"])),
            preprocessing_sha256=Sha256(str(raw["preprocessing_sha256"])),
            model_file_sha256=Sha256(str(raw["model_file_sha256"])),
            deep_svdd_center_sha256=(
                None
                if raw["deep_svdd_center_sha256"] is None
                else Sha256(str(raw["deep_svdd_center_sha256"]))
            ),
            training_rows=tuple(
                ClientTrainingCount(
                    ClientId(str(item["client_id"])),
                    int(item["rows"]),
                )
                for item in raw["training_rows"]
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
                client_id=ClientId(str(item["client_id"])),
                mean_loss=float(item["mean_loss"]),
                record_presentations=int(item["record_presentations"]),
                optimizer_steps=int(item["optimizer_steps"]),
                model_hash=Sha256(str(item["model_hash"])),
            )
            for item in raw["client_results"]
        )
        return RoundResult(
            round_index=int(raw["round_index"]),
            learning_rate=float(raw["learning_rate"]),
            selected_clients=tuple(
                ClientId(str(value)) for value in raw["selected_clients"]
            ),
            client_results=clients,
            mean_client_loss=float(raw["mean_client_loss"]),
            minimum_client_loss=float(raw["minimum_client_loss"]),
            maximum_client_loss=float(raw["maximum_client_loss"]),
            parameter_update_norm=float(raw["parameter_update_norm"]),
            model_payload_bytes=int(raw["model_payload_bytes"]),
            round_communication_bytes=int(raw["round_communication_bytes"]),
            global_model_hash=Sha256(str(raw["global_model_hash"])),
        )
