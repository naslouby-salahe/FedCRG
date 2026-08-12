"""Frozen detector training and immutable training-spec model-cache persistence."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from fedcrg.artifacts.hashing import sha256_file
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.constants import (
    DIAD_AUTOENCODER_BYTES,
    DIAD_AUTOENCODER_PARAMETERS,
    NBAIOT_AUTOENCODER_BYTES,
    NBAIOT_AUTOENCODER_PARAMETERS,
)
from fedcrg.core.enums import DataRole, DatasetId, DetectorId, ExperimentId, FailureCode
from fedcrg.core.ids import ClientId
from fedcrg.detectors.base import DetectorModel
from fedcrg.detectors.deep_svdd import DeepSvdd
from fedcrg.detectors.factory import DetectorFactory
from fedcrg.federated.trainer import FederatedTrainer

_METADATA_COLUMNS = {
    "row_id",
    "role",
    "label",
    "attack_group",
    "source_file",
    "source_row_index",
    "capture_time",
}


def feature_columns(frame: pd.DataFrame, expected_count: int) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column not in _METADATA_COLUMNS
        and not column.startswith("_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(columns) != expected_count:
        raise ValueError(
            f"{FailureCode.FEATURE_SCHEMA_MISMATCH.value}: expected {expected_count} "
            f"numeric features, found {len(columns)}"
        )
    return columns


class TrainDetector:
    """Train one frozen detector per training spec/model seed and never policy-tune it."""

    def __init__(
        self,
        factory: DetectorFactory | None = None,
        trainer: FederatedTrainer | None = None,
    ) -> None:
        self.factory = factory or DetectorFactory()
        self.trainer = trainer or FederatedTrainer()

    def create_model(self, config: ExperimentConfig) -> DetectorModel:
        return self.factory.create(config.dataset.feature_count, config.detector)

    def _validate_architecture(self, config: ExperimentConfig, model: DetectorModel) -> None:
        if config.detector.id is not DetectorId.AUTOENCODER:
            return
        if config.id is ExperimentId.DIAD_FEATURE_SENSITIVITY:
            dimension = config.dataset.feature_count
            expected_hidden = (
                max(1, int(0.75 * dimension)),
                max(1, int(0.50 * dimension)),
                max(1, dimension // 3),
                max(1, int(0.25 * dimension)),
            )
            if tuple(config.detector.hidden_dims) != expected_hidden:
                raise RuntimeError(
                    f"R14 architecture rule failed: {config.detector.hidden_dims} != {expected_hidden}"
                )
            return

        expected_parameters = (
            NBAIOT_AUTOENCODER_PARAMETERS
            if config.dataset.id is DatasetId.NBAIOT
            else DIAD_AUTOENCODER_PARAMETERS
        )
        expected_bytes = (
            NBAIOT_AUTOENCODER_BYTES
            if config.dataset.id is DatasetId.NBAIOT
            else DIAD_AUTOENCODER_BYTES
        )
        if model.trainable_parameter_count() != expected_parameters:
            raise RuntimeError(
                "Detector parameter-count contract failed: "
                f"{model.trainable_parameter_count()} != {expected_parameters}"
            )
        if model.trainable_tensor_bytes() != expected_bytes:
            raise RuntimeError(
                "Detector tensor-byte contract failed: "
                f"{model.trainable_tensor_bytes()} != {expected_bytes}"
            )

    def train_from_cache(
        self,
        config: ExperimentConfig,
        prepared_root: Path,
        model_seed: int,
    ) -> tuple[Path, Path]:
        if model_seed not in config.randomness.model_seeds:
            raise ValueError(f"Model seed {model_seed} is not configured")

        prepared_manifest_path = prepared_root / "manifest.json"
        prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
        if prepared_manifest.get("data_spec_hash") != config.data_spec_hash:
            raise ValueError("Prepared dataset data-spec hash does not match training config")

        datasets: dict[ClientId, TensorDataset] = {}
        training_rows: dict[str, int] = {}
        for client_value in sorted(prepared_manifest["clients"]):
            client_id = ClientId(client_value)
            path = prepared_root / "clients" / client_value / f"{DataRole.TRAIN.value}.csv.gz"
            frame = pd.read_csv(path)
            if len(frame) != config.dataset.split.train_benign:
                raise ValueError(
                    f"Training-row contract failed for {client_id}: {len(frame)} != "
                    f"{config.dataset.split.train_benign}"
                )
            columns = feature_columns(frame, config.dataset.feature_count)
            tensor = torch.as_tensor(
                frame[columns].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(FailureCode.TRAINING_NUMERICAL_FAILURE.value)
            datasets[client_id] = TensorDataset(tensor)
            training_rows[client_value] = len(frame)

        model_root = (
            config.outputs_root
            / "cache"
            / "models"
            / config.dataset.id.value
            / config.detector.id.value
            / f"m{model_seed}"
            / config.training_spec_hash[:16]
        )
        model_path = model_root / "model.pt"
        manifest_path = model_root / "training.json"
        if model_path.exists() or manifest_path.exists():
            self._validate_existing_cache(config, model_path, manifest_path)
            return model_path, manifest_path

        torch.manual_seed(model_seed)
        model = self.create_model(config)
        self._validate_architecture(config, model)
        center_hash: str | None = None
        if isinstance(model, DeepSvdd):
            # The center is the equal mean of client initial embeddings and is then frozen.
            model.initialize_center([dataset.tensors[0] for dataset in datasets.values()])
            center_hash = hashlib.sha256(
                model.center.detach().cpu().numpy().tobytes()
            ).hexdigest()

        final_model, result = self.trainer.train(
            model,
            datasets,
            config.training,
            model_seed,
        )
        model_root.mkdir(parents=True, exist_ok=False)
        temp_model = model_root / ".model.pt.tmp"
        torch.save(final_model.state_dict(), temp_model)
        os.replace(temp_model, model_path)

        payload = {
            **asdict(result),
            "dataset_id": config.dataset.id.value,
            "detector_id": config.detector.id.value,
            "data_spec_hash": config.data_spec_hash,
            "training_spec_hash": config.training_spec_hash,
            "dataset_manifest_sha256": sha256_file(prepared_manifest_path),
            "preprocessing_sha256": sha256_file(prepared_root / "preprocessing.json"),
            "model_file_sha256": sha256_file(model_path),
            "training_rows": training_rows,
            "deep_svdd_center_sha256": center_hash,
        }
        temp_manifest = model_root / ".training.json.tmp"
        temp_manifest.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temp_manifest, manifest_path)
        return model_path, manifest_path

    @staticmethod
    def _validate_existing_cache(
        config: ExperimentConfig,
        model_path: Path,
        manifest_path: Path,
    ) -> None:
        if not model_path.is_file() or not manifest_path.is_file():
            raise FileExistsError("Model cache is partially materialized")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("training_spec_hash") != config.training_spec_hash:
            raise ValueError("Existing model cache belongs to another training specification")
        if manifest.get("model_file_sha256") != sha256_file(model_path):
            raise ValueError("Existing frozen-model hash does not match its manifest")

    def load_model(self, config: ExperimentConfig, model_path: Path) -> DetectorModel:
        model = self.create_model(config)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self._validate_architecture(config, model)
        return model
