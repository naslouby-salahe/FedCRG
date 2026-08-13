"""Atomic, hash-finalized Parquet score-cache persistence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.artifacts.hashing import sha256_file
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.core.enums import DataRole, DatasetId
from fedcrg.core.exceptions import ImmutableRunError
from fedcrg.core.ids import ClientId, Sha256
from fedcrg.scoring.integrity import validate_score_manifest
from fedcrg.scoring.models import ClientScoreSet, RoleScores, ScoreManifest


class ScoreCache:
    filename = "score_cache.parquet"
    manifest_filename = "manifest.json"

    def save(self, manifest: ScoreManifest, root: Path) -> ScoreManifest:
        validate_score_manifest(manifest)
        root.mkdir(parents=True, exist_ok=True)
        parquet_path = root / self.filename
        metadata_path = root / self.manifest_filename
        if parquet_path.exists() or metadata_path.exists():
            raise ImmutableRunError(f"Score cache already exists and is immutable: {root}")

        records: list[dict[str, object]] = []
        for client_id in sorted(manifest.clients):
            client = manifest.clients[client_id]
            for role in sorted(client.scores, key=lambda item: item.value):
                scores = client.scores[role]
                groups = scores.attack_groups or (None,) * len(scores.values)
                for row_id, score, group in zip(
                    scores.row_ids,
                    scores.values,
                    groups,
                    strict=True,
                ):
                    records.append(
                        {
                            "dataset_id": manifest.dataset.value,
                            "client_id": client_id.value,
                            "row_id": row_id,
                            "phase": role.value,
                            "model_seed": manifest.model_seed,
                            "score_float64": float(score),
                            "label_test_only": (0 if role is DataRole.BENIGN_TEST else 1 if role is DataRole.ATTACK_TEST else None),
                            "attack_family_test_only": (group if role is DataRole.ATTACK_TEST else None),
                        }
                    )
        frame = pd.DataFrame.from_records(records)
        frame["score_float64"] = frame["score_float64"].astype(np.float64)
        temp = parquet_path.with_suffix(".parquet.tmp")
        frame.to_parquet(temp, index=False, engine="pyarrow")
        temp.replace(parquet_path)
        cache_hash = Sha256(sha256_file(parquet_path))

        finalized = replace(manifest, cache_sha256=cache_hash)
        atomic_write_json(
            metadata_path,
            {
                "dataset": finalized.dataset.value,
                "model_seed": finalized.model_seed,
                "model_hash": finalized.model_hash.value,
                "data_spec_hash": finalized.data_spec_hash.value,
                "training_spec_hash": finalized.training_spec_hash.value,
                "dataset_manifest_hash": finalized.dataset_manifest_hash.value,
                "preprocessing_hash": finalized.preprocessing_hash.value,
                "score_cache_file": self.filename,
                "score_cache_sha256": cache_hash.value,
                "role_hashes": finalized.role_hashes(),
            },
        )
        return finalized

    def load(self, root: Path) -> ScoreManifest:
        metadata_path = root / self.manifest_filename
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parquet_path = root / str(metadata["score_cache_file"])
        expected_hash = Sha256(str(metadata["score_cache_sha256"]))
        actual_hash = Sha256(sha256_file(parquet_path))
        if actual_hash != expected_hash:
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: serialized cache hash differs")

        frame = pd.read_parquet(parquet_path, engine="pyarrow")
        clients: dict[ClientId, ClientScoreSet] = {}
        for client_id, client_frame in frame.groupby("client_id", sort=True):
            score_map = {}
            for role_value, role_frame in client_frame.groupby("phase", sort=True):
                role = DataRole(str(role_value))
                group_values = role_frame["attack_family_test_only"]
                groups = None
                if group_values.notna().any():
                    groups = tuple(group_values.fillna("").astype(str))
                role_scores = RoleScores(
                    role=role,
                    values=role_frame["score_float64"].to_numpy(dtype=np.float64),
                    client_id=ClientId(str(client_id)),
                    row_ids=tuple(role_frame["row_id"].astype(str)),
                    attack_groups=groups,
                )
                expected_role_hash = metadata["role_hashes"][str(client_id)][role.value]
                if role_scores.sha256.value != expected_role_hash:
                    raise ValueError(
                        f"SCORE_CACHE_HASH_MISMATCH: {client_id}/{role.value}"
                    )
                score_map[role] = role_scores
            typed_client_id = ClientId(str(client_id))
            clients[typed_client_id] = ClientScoreSet(typed_client_id, score_map)

        manifest = ScoreManifest(
            dataset=DatasetId(metadata["dataset"]),
            model_seed=int(metadata["model_seed"]),
            model_hash=Sha256(str(metadata["model_hash"])),
            data_spec_hash=Sha256(str(metadata["data_spec_hash"])),
            training_spec_hash=Sha256(str(metadata["training_spec_hash"])),
            dataset_manifest_hash=Sha256(str(metadata["dataset_manifest_hash"])),
            preprocessing_hash=Sha256(str(metadata["preprocessing_hash"])),
            clients=clients,
            cache_sha256=actual_hash,
        )
        validate_score_manifest(manifest)
        return manifest
