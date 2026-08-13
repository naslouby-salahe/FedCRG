"""Frozen detector training and immutable training-spec model-cache persistence."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from fedcrg.artifacts.dataset import PreparedDatasetManifestStore
from fedcrg.artifacts.hashing import sha256_file
from fedcrg.artifacts.training import (
    ClientTrainingCount,
    TrainingManifest,
    TrainingManifestStore,
)
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.constants import (
    DIAD_AUTOENCODER_BYTES,
    DIAD_AUTOENCODER_PARAMETERS,
    NBAIOT_AUTOENCODER_BYTES,
    NBAIOT_AUTOENCODER_PARAMETERS,
)
from fedcrg.core.enums import DataRole, DatasetId, DetectorId, ExperimentId, FailureCode
from fedcrg.core.ids import ClientId, ModelSeed, Sha256
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
        manifests: TrainingManifestStore | None = None,
        dataset_manifests: PreparedDatasetManifestStore | None = None,
    ) -> None:
        self.factory = factory or DetectorFactory()
        self.trainer = trainer or FederatedTrainer()
        self.manifests = manifests or TrainingManifestStore()
        self.dataset_manifests = dataset_manifests or PreparedDatasetManifestStore()

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
        model_seed: ModelSeed | int,
    ) -> tuple[Path, Path]:
        seed = ModelSeed(int(model_seed))
        if int(seed) not in config.randomness.model_seeds:
            raise ValueError(f"Model seed {int(seed)} is not configured")

        prepared_manifest_path = prepared_root / "manifest.json"
        preprocessing_path = prepared_root / "preprocessing.json"
        prepared_manifest = self.dataset_manifests.load(prepared_manifest_path)
        if prepared_manifest.data_spec_hash != Sha256(config.data_spec_hash):
            raise ValueError("Prepared dataset data-spec hash does not match training config")
        if prepared_manifest.dataset_id is not config.dataset.id:
            raise ValueError("Prepared dataset identity does not match training config")
        prepared_manifest_hash = Sha256(sha256_file(prepared_manifest_path))
        preprocessing_hash = Sha256(sha256_file(preprocessing_path))

        datasets: dict[ClientId, TensorDataset] = {}
        training_rows: list[ClientTrainingCount] = []
        for client_manifest in prepared_manifest.clients:
            client_id = client_manifest.client_id
            train_role = client_manifest.role(DataRole.TRAIN)
            path = prepared_root / Path(train_role.relative_path)
            if Sha256(sha256_file(path)) != train_role.file_sha256:
                raise ValueError(f"Prepared training file hash mismatch for {client_id.value}")
            frame = pd.read_csv(path)
            if len(frame) != config.dataset.split.train_benign:
                raise ValueError(
                    f"Training-row contract failed for {client_id}: {len(frame)} != "
                    f"{config.dataset.split.train_benign}"
                )
            columns = feature_columns(frame, config.dataset.feature_count)
            if tuple(columns) != prepared_manifest.feature_names:
                raise ValueError("Training feature order differs from frozen dataset manifest")
            tensor = torch.as_tensor(
                frame[columns].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(FailureCode.TRAINING_NUMERICAL_FAILURE.value)
            datasets[client_id] = TensorDataset(tensor)
            training_rows.append(ClientTrainingCount(client_id, len(frame)))

        model_root = (
            config.outputs_root
            / "cache"
            / "models"
            / config.dataset.id.value
            / config.detector.id.value
            / f"m{int(seed)}"
            / config.training_spec_hash[:16]
        )
        model_path = model_root / "model.pt"
        manifest_path = model_root / "training.json"
        if model_path.exists() or manifest_path.exists():
            self._validate_existing_cache(
                config,
                seed,
                model_path,
                manifest_path,
                prepared_manifest_hash,
                preprocessing_hash,
            )
            return model_path, manifest_path

        torch.manual_seed(int(seed))
        model = self.create_model(config)
        self._validate_architecture(config, model)
        center_hash: Sha256 | None = None
        if isinstance(model, DeepSvdd):
            model.initialize_center([dataset.tensors[0] for dataset in datasets.values()])
            center_hash = Sha256(
                hashlib.sha256(model.center.detach().cpu().numpy().tobytes()).hexdigest()
            )

        final_model, result = self.trainer.train(
            model,
            datasets,
            config.training,
            seed,
        )
        model_root.mkdir(parents=True, exist_ok=False)
        temp_model = model_root / ".model.pt.tmp"
        torch.save(final_model.state_dict(), temp_model)
        os.replace(temp_model, model_path)

        manifest = TrainingManifest(
            dataset_id=config.dataset.id,
            detector_id=config.detector.id,
            model_seed=seed,
            data_spec_hash=Sha256(config.data_spec_hash),
            training_spec_hash=Sha256(config.training_spec_hash),
            dataset_manifest_sha256=prepared_manifest_hash,
            preprocessing_sha256=preprocessing_hash,
            model_file_sha256=Sha256(sha256_file(model_path)),
            deep_svdd_center_sha256=center_hash,
            training_rows=tuple(training_rows),
            result=result,
        )
        self.manifests.save(manifest_path, manifest)
        return model_path, manifest_path

    def _validate_existing_cache(
        self,
        config: ExperimentConfig,
        model_seed: ModelSeed,
        model_path: Path,
        manifest_path: Path,
        prepared_manifest_hash: Sha256,
        preprocessing_hash: Sha256,
    ) -> None:
        if not model_path.is_file() or not manifest_path.is_file():
            raise FileExistsError("Model cache is partially materialized")
        manifest = self.manifests.load(manifest_path)
        if manifest.model_seed != model_seed:
            raise ValueError("Existing model cache has a different model seed")
        if manifest.data_spec_hash != Sha256(config.data_spec_hash):
            raise ValueError("Existing model cache belongs to another data specification")
        if manifest.training_spec_hash != Sha256(config.training_spec_hash):
            raise ValueError("Existing model cache belongs to another training specification")
        if manifest.dataset_manifest_sha256 != prepared_manifest_hash:
            raise ValueError("Existing model cache was trained from a different prepared manifest")
        if manifest.preprocessing_sha256 != preprocessing_hash:
            raise ValueError(
                "Existing model cache was trained with different preprocessing evidence"
            )
        if manifest.model_file_sha256 != Sha256(sha256_file(model_path)):
            raise ValueError("Existing frozen-model hash does not match its manifest")
        if manifest.result.final_model_hash != Sha256(
            self.load_model(config, model_path).state_hash()
        ):
            raise ValueError("Existing model state hash does not match training result")

    def load_model(self, config: ExperimentConfig, model_path: Path) -> DetectorModel:
        model = self.create_model(config)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self._validate_architecture(config, model)
        return model
