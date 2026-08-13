"""Typed anomaly-score inputs and immutable score-cache contents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from fedcrg.core.enums import DataRole, DatasetId
from fedcrg.core.ids import ClientId, Sha256


@dataclass(frozen=True, slots=True)
class RoleScoreInput:
    role: DataRole
    values: np.ndarray
    row_ids: tuple[str, ...]
    attack_groups: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError("Detector inputs must be a two-dimensional feature matrix")
        if len(values) != len(self.row_ids):
            raise ValueError("row_ids must align with detector inputs")
        if self.attack_groups is not None and len(values) != len(self.attack_groups):
            raise ValueError("attack_groups must align with detector inputs")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class ClientScoreInput:
    client_id: ClientId
    roles: dict[DataRole, RoleScoreInput]


@dataclass(frozen=True, slots=True)
class RoleScores:
    role: DataRole
    values: np.ndarray
    client_id: ClientId
    row_ids: tuple[str, ...]
    attack_groups: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("Role scores must be one-dimensional")
        if len(values) != len(self.row_ids):
            raise ValueError("Row provenance must align with score values")
        if self.attack_groups is not None and len(self.attack_groups) != len(values):
            raise ValueError("Attack-group metadata must align with score values")
        if not np.isfinite(values).all():
            raise ValueError("NONFINITE_SCORE: cached anomaly scores must be finite")
        object.__setattr__(self, "values", values)

    @property
    def sha256(self) -> Sha256:
        digest = hashlib.sha256()
        digest.update(self.role.value.encode("utf-8"))
        digest.update(self.client_id.value.encode("utf-8"))
        for row_id, value in zip(self.row_ids, self.values, strict=True):
            digest.update(row_id.encode("ascii"))
            digest.update(np.float64(value).tobytes())
        if self.attack_groups is not None:
            for group in self.attack_groups:
                digest.update(group.encode("utf-8"))
        return Sha256(digest.hexdigest())


@dataclass(frozen=True, slots=True)
class ClientScoreSet:
    client_id: ClientId
    scores: dict[DataRole, RoleScores]


@dataclass(frozen=True, slots=True)
class ScoreManifest:
    dataset: DatasetId
    model_seed: int
    model_hash: Sha256
    data_spec_hash: Sha256
    training_spec_hash: Sha256
    dataset_manifest_hash: Sha256
    preprocessing_hash: Sha256
    clients: dict[ClientId, ClientScoreSet]
    cache_sha256: Sha256 | None = None

    def role_hashes(self) -> dict[str, dict[str, str]]:
        return {
            client_id.value: {
                role.value: role_scores.sha256.value
                for role, role_scores in client.scores.items()
            }
            for client_id, client in self.clients.items()
        }
