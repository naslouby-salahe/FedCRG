"""Frozen-model anomaly-score computation."""

from __future__ import annotations

import numpy as np
import torch

from fedcrg.core.enums import ComputeDeviceId, DataRole, DatasetId
from fedcrg.core.ids import ClientId, ModelSeed, Sha256
from fedcrg.detectors.base import DetectorModel
from fedcrg.scoring.models import ClientScoreInput, ClientScoreSet, RoleScores, ScoreManifest


class ScoreComputer:
    """Compute scores without exposing policy or calibration-seed concepts."""

    def compute(
        self,
        model: DetectorModel,
        values: np.ndarray,
        device: ComputeDeviceId = ComputeDeviceId.CPU,
        batch_size: int = 65_536,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("Scoring batch_size must be positive")
        torch_device = torch.device(device.value)
        model = model.to(torch_device).eval()
        result: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                batch = torch.as_tensor(
                    values[start : start + batch_size],
                    dtype=torch.float32,
                    device=torch_device,
                )
                scores = model.anomaly_score(batch).detach().cpu().numpy()
                result.append(scores.astype(np.float64, copy=False))
        return np.concatenate(result) if result else np.empty(0, dtype=np.float64)

    def compute_manifest(
        self,
        model: DetectorModel,
        dataset: DatasetId,
        model_seed: ModelSeed,
        data_spec_hash: Sha256,
        training_spec_hash: Sha256,
        dataset_manifest_hash: Sha256,
        preprocessing_hash: Sha256,
        clients: tuple[ClientScoreInput, ...],
        device: ComputeDeviceId = ComputeDeviceId.CPU,
    ) -> ScoreManifest:
        scored_clients: dict[ClientId, ClientScoreSet] = {}
        for client in clients:
            role_scores: dict[DataRole, RoleScores] = {}
            for role, role_input in client.roles.items():
                role_scores[role] = RoleScores(
                    role=role,
                    values=self.compute(model, role_input.values, device),
                    client_id=client.client_id,
                    row_ids=role_input.row_ids,
                    attack_groups=role_input.attack_groups,
                )
            scored_clients[client.client_id] = ClientScoreSet(client.client_id, role_scores)
        return ScoreManifest(
            dataset=dataset,
            model_seed=model_seed,
            model_hash=Sha256(model.state_hash()),
            data_spec_hash=data_spec_hash,
            training_spec_hash=training_spec_hash,
            dataset_manifest_hash=dataset_manifest_hash,
            preprocessing_hash=preprocessing_hash,
            clients=scored_clients,
        )
