"""Detector construction, federated training, and model-cache persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from fedcrg.config.models import ExperimentConfig
from fedcrg.detectors.base import DetectorModel
from fedcrg.detectors.deep_svdd import DeepSvdd
from fedcrg.detectors.factory import DetectorFactory
from fedcrg.federated.trainer import FederatedTrainer

_METADATA_COLUMNS = {"row_id", "role", "label", "attack_group"}


def feature_columns(frame: pd.DataFrame, expected_count: int) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column not in _METADATA_COLUMNS
        and not column.startswith("_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if len(columns) != expected_count:
        raise ValueError(f"Expected {expected_count} numeric features, found {len(columns)}")
    return columns


class TrainDetector:
    def __init__(self, factory: DetectorFactory | None = None, trainer: FederatedTrainer | None = None) -> None:
        self.factory = factory or DetectorFactory()
        self.trainer = trainer or FederatedTrainer()

    def create_model(self, config: ExperimentConfig) -> DetectorModel:
        return self.factory.create(config.dataset.feature_count, config.detector)

    def train_from_cache(
        self,
        config: ExperimentConfig,
        prepared_root: Path,
        model_seed: int,
    ) -> tuple[Path, Path]:
        if model_seed not in config.randomness.model_seeds:
            raise ValueError(f"Model seed {model_seed} is not configured")
        manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
        datasets: dict[str, TensorDataset] = {}
        for client_id in sorted(manifest["clients"]):
            frame = pd.read_csv(prepared_root / client_id / "train.csv.gz")
            columns = feature_columns(frame, config.dataset.feature_count)
            tensor = torch.as_tensor(frame[columns].to_numpy(), dtype=torch.float32)
            datasets[client_id] = TensorDataset(tensor)
        torch.manual_seed(model_seed)
        model = self.create_model(config)
        if isinstance(model, DeepSvdd):
            batches = [dataset.tensors[0] for dataset in datasets.values()]
            model.initialize_center(batches)
        final_model, result = self.trainer.train(model, datasets, config.training, model_seed)
        model_root = (
            config.outputs_root
            / "cache"
            / "models"
            / config.dataset.id.value
            / config.detector.id.value
            / f"m{model_seed}"
            / config.config_hash[:16]
        )
        model_root.mkdir(parents=True, exist_ok=True)
        model_path = model_root / "model.pt"
        temp_model = model_root / "model.pt.tmp"
        torch.save(final_model.state_dict(), temp_model)
        os.replace(temp_model, model_path)
        manifest_path = model_root / "training.json"
        temp_manifest = model_root / "training.json.tmp"
        temp_manifest.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
        os.replace(temp_manifest, manifest_path)
        return model_path, manifest_path

    def load_model(self, config: ExperimentConfig, model_path: Path) -> DetectorModel:
        model = self.create_model(config)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model
