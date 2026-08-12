"""Anomaly-score computation."""

from __future__ import annotations

import numpy as np
import torch

from fedcrg.core.enums import DataRole, DatasetId
from fedcrg.detectors.base import DetectorModel
from fedcrg.scoring.models import ClientScoreSet, RoleScores, ScoreManifest


class ScoreComputer:
    def compute(self, model: DetectorModel, values: np.ndarray, device: str = "cpu") -> np.ndarray:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=torch.device(device))
        model = model.to(device).eval()
        with torch.no_grad():
            scores = model.anomaly_score(tensor).detach().cpu().numpy().astype(np.float64, copy=False)
        return scores

    def compute_manifest(self, model: DetectorModel, dataset: DatasetId, model_seed: int, role_values: dict[str, dict[DataRole, np.ndarray]], role_groups: dict[str, dict[DataRole, tuple[str, ...]]] | None = None, device: str = "cpu") -> ScoreManifest:
        groups = role_groups or {}
        clients: dict[str, ClientScoreSet] = {}
        for client_id, roles in role_values.items():
            scores = {role: RoleScores(role, self.compute(model, values, device), client_id, groups.get(client_id, {}).get(role)) for role, values in roles.items()}
            clients[client_id] = ClientScoreSet(client_id, scores)
        return ScoreManifest(dataset, model_seed, model.state_hash(), clients)
