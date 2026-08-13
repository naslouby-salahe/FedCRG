"""Typed, persistence-facing anomaly-score cache record."""

from __future__ import annotations

from dataclasses import dataclass

from fedcrg.domain.enums import DatasetId
from fedcrg.domain.identifiers import ClientId, ModelSeed, Sha256
from fedcrg.scoring.calibration_scores import ClientScoreSet


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
