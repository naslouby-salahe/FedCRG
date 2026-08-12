"""Atomic score cache persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fedcrg.core.enums import DataRole, DatasetId
from fedcrg.scoring.integrity import validate_score_manifest
from fedcrg.scoring.models import ClientScoreSet, RoleScores, ScoreManifest


class ScoreCache:
    def save(self, manifest: ScoreManifest, root: Path) -> None:
        validate_score_manifest(manifest)
        root.mkdir(parents=True, exist_ok=True)
        metadata = {"dataset": manifest.dataset.value, "model_seed": manifest.model_seed, "model_hash": manifest.model_hash, "clients": {}}
        for client_id, client in manifest.clients.items():
            client_meta = {}
            for role, scores in client.scores.items():
                path = root / f"{client_id}__{role.value}.npy"
                temp = path.with_suffix(".npy.tmp")
                with temp.open("wb") as handle:
                    np.save(handle, scores.values, allow_pickle=False)
                temp.replace(path)
                client_meta[role.value] = {"file": path.name, "sha256": scores.sha256, "attack_groups": list(scores.attack_groups) if scores.attack_groups is not None else None}
            metadata["clients"][client_id] = client_meta
        temp_manifest = root / "manifest.json.tmp"
        temp_manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        temp_manifest.replace(root / "manifest.json")

    def load(self, root: Path) -> ScoreManifest:
        metadata = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        clients: dict[str, ClientScoreSet] = {}
        for client_id, roles in metadata["clients"].items():
            score_map = {}
            for role_value, item in roles.items():
                role = DataRole(role_value)
                values = np.load(root / item["file"], allow_pickle=False).astype(np.float64, copy=False)
                groups = item.get("attack_groups")
                role_scores = RoleScores(role, values, client_id, tuple(str(value) for value in groups) if groups is not None else None)
                if role_scores.sha256 != item["sha256"]:
                    raise ValueError(f"Score hash mismatch for {client_id}/{role.value}")
                score_map[role] = role_scores
            clients[client_id] = ClientScoreSet(client_id, score_map)
        manifest = ScoreManifest(dataset=DatasetId(metadata["dataset"]), model_seed=int(metadata["model_seed"]), model_hash=str(metadata["model_hash"]), clients=clients)
        validate_score_manifest(manifest)
        return manifest
