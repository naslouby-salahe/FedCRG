"""Typed anomaly-score inputs and immutable score-cache contents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from fedcrg.domain.enums import DataRole, DatasetId
from fedcrg.domain.identifiers import AttackGroupId, ClientId, ModelSeed, RowId, Sha256


@dataclass(frozen=True, slots=True)
class RoleScoreInput:
    role: DataRole
    values: np.ndarray
    row_ids: tuple[RowId, ...]
    attack_groups: tuple[AttackGroupId, ...] | None = None

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
    roles: tuple[RoleScoreInput, ...]

    def get(self, role: DataRole) -> RoleScoreInput:
        for item in self.roles:
            if item.role is role:
                return item
        raise KeyError(role.value)


@dataclass(frozen=True, slots=True)
class RoleScores:
    role: DataRole
    values: np.ndarray
    client_id: ClientId
    row_ids: tuple[RowId, ...]
    attack_groups: tuple[AttackGroupId, ...] | None = None

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
            digest.update(row_id.value.encode("ascii"))
            digest.update(np.float64(value).tobytes())
        if self.attack_groups is not None:
            for group in self.attack_groups:
                digest.update(group.value.encode("utf-8"))
        return Sha256(digest.hexdigest())


@dataclass(frozen=True, slots=True)
class ClientScoreSet:
    client_id: ClientId
    scores: tuple[RoleScores, ...]

    def get(self, role: DataRole) -> RoleScores:
        for item in self.scores:
            if item.role is role:
                return item
        raise KeyError(role.value)


@dataclass(frozen=True, slots=True)
class ScoreManifest:
    dataset: DatasetId
    model_seed: ModelSeed
    model_hash: Sha256
    data_spec_hash: Sha256
    training_spec_hash: Sha256
    dataset_manifest_hash: Sha256
    preprocessing_hash: Sha256
    clients: tuple[ClientScoreSet, ...]
    cache_sha256: Sha256 | None = None

    def client(self, client_id: ClientId) -> ClientScoreSet:
        for item in self.clients:
            if item.client_id == client_id:
                return item
        raise KeyError(client_id.value)

    def role_hashes(self) -> dict[str, dict[str, str]]:
        return {
            client.client_id.value: {
                role_scores.role.value: role_scores.sha256.value for role_scores in client.scores
            }
            for client in self.clients
        }
