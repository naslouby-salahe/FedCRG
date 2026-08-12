"""Typed anomaly-score containers with evaluation metadata."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from fedcrg.core.enums import DataRole, DatasetId


@dataclass(frozen=True, slots=True)
class RoleScores:
    role: DataRole
    values: np.ndarray
    client_id: str
    attack_groups: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("Role scores must be one-dimensional")
        if self.attack_groups is not None and len(self.attack_groups) != len(values):
            raise ValueError("Attack-group metadata must align with score values")
        object.__setattr__(self, "values", values)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.values.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ClientScoreSet:
    client_id: str
    scores: dict[DataRole, RoleScores]


@dataclass(frozen=True, slots=True)
class ScoreManifest:
    dataset: DatasetId
    model_seed: int
    model_hash: str
    clients: dict[str, ClientScoreSet]

    def role_hashes(self) -> dict[str, dict[str, str]]:
        return {client_id: {role.value: scores.sha256 for role, scores in client.scores.items()} for client_id, client in self.clients.items()}
