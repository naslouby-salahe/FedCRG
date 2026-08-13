"""Atomic, hash-finalized Parquet score-cache persistence.

The writer is streaming: only one scored data role needs to be resident in memory at
once. This matters for DIAD and the large N-BaIoT attack files, where constructing a
Python record for every cached score before serialization is unnecessarily expensive.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from fedcrg.artifacts.hashing import sha256_file
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.core.enums import DataRole, DatasetId
from fedcrg.core.exceptions import ImmutableRunError
from fedcrg.core.ids import AttackGroupId, ClientId, ModelSeed, RowId, Sha256
from fedcrg.scoring.integrity import validate_score_manifest
from fedcrg.scoring.models import ClientScoreSet, RoleScores, ScoreManifest


@dataclass(frozen=True, slots=True)
class ScoreCacheIdentity:
    """Provenance that is fixed before score serialization starts."""

    dataset: DatasetId
    model_seed: ModelSeed
    model_hash: Sha256
    data_spec_hash: Sha256
    training_spec_hash: Sha256
    dataset_manifest_hash: Sha256
    preprocessing_hash: Sha256


@dataclass(frozen=True, slots=True)
class ScoreCacheDescriptor:
    """Small metadata object returned after a streaming cache is finalized."""

    identity: ScoreCacheIdentity
    cache_sha256: Sha256
    client_ids: tuple[ClientId, ...]
    role_hashes: dict[str, dict[str, str]]
    role_counts: dict[str, dict[str, int]]


class ScoreCache:
    filename = "score_cache.parquet"
    manifest_filename = "manifest.json"

    def save(self, manifest: ScoreManifest, root: Path) -> ScoreManifest:
        """Persist an already-materialized manifest used by small fixtures."""

        validate_score_manifest(manifest)
        descriptor = self.save_stream(
            ScoreCacheIdentity(
                dataset=manifest.dataset,
                model_seed=manifest.model_seed,
                model_hash=manifest.model_hash,
                data_spec_hash=manifest.data_spec_hash,
                training_spec_hash=manifest.training_spec_hash,
                dataset_manifest_hash=manifest.dataset_manifest_hash,
                preprocessing_hash=manifest.preprocessing_hash,
            ),
            (
                role_scores
                for client_id in sorted(manifest.clients)
                for role_scores in manifest.clients[client_id].scores.values()
            ),
            root,
        )
        return replace(manifest, cache_sha256=descriptor.cache_sha256)

    def save_stream(
        self,
        identity: ScoreCacheIdentity,
        role_scores: Iterable[RoleScores],
        root: Path,
    ) -> ScoreCacheDescriptor:
        """Write role-score blocks incrementally and finalize one immutable cache."""

        if root.exists():
            raise ImmutableRunError(f"Score cache already exists and is immutable: {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = root.parent / f".{root.name}.staging-{uuid.uuid4().hex}"
        staging.mkdir()
        parquet_path = staging / self.filename
        writer = None
        role_hashes: dict[str, dict[str, str]] = {}
        role_counts: dict[str, dict[str, int]] = {}
        observed_roles: dict[ClientId, set[DataRole]] = {}

        try:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:  # pragma: no cover - dependency/runtime boundary
                raise RuntimeError(
                    "Parquet score-cache serialization requires the declared pyarrow dependency"
                ) from exc

            for scores in role_scores:
                if scores.role in {
                    DataRole.REFERENCE,
                    DataRole.MISMATCH,
                    DataRole.CALIBRATION,
                    DataRole.BENIGN_GUARD,
                }:
                    raise ValueError(
                        "Calibration-assignment roles are views and must not be serialized in the physical score cache"
                    )
                cid = scores.client_id
                observed_roles.setdefault(cid, set())
                if scores.role in observed_roles[cid]:
                    raise ValueError(f"Duplicate score block: {cid.value}/{scores.role.value}")
                observed_roles[cid].add(scores.role)

                frame = self._role_frame(identity, scores)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        parquet_path,
                        table.schema,
                        compression="zstd",
                        use_dictionary=True,
                        write_statistics=True,
                    )
                writer.write_table(table)
                role_hashes.setdefault(cid.value, {})[scores.role.value] = scores.sha256.value
                role_counts.setdefault(cid.value, {})[scores.role.value] = len(scores.values)
                del table, frame

            if writer is None:
                raise ValueError("Cannot finalize an empty score cache")
            writer.close()
            writer = None
            self._validate_stream_roles(observed_roles)
            cache_hash = Sha256(sha256_file(parquet_path))
            client_ids = tuple(sorted(observed_roles))
            atomic_write_json(
                staging / self.manifest_filename,
                self._metadata_payload(
                    identity,
                    cache_hash,
                    client_ids,
                    role_hashes,
                    role_counts,
                ),
            )
            os.replace(staging, root)
            return ScoreCacheDescriptor(
                identity=identity,
                cache_sha256=cache_hash,
                client_ids=client_ids,
                role_hashes=role_hashes,
                role_counts=role_counts,
            )
        except Exception:
            if writer is not None:
                writer.close()
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _role_frame(identity: ScoreCacheIdentity, scores: RoleScores) -> pd.DataFrame:
        groups = scores.attack_groups or (None,) * len(scores.values)
        label: int | None
        if scores.role is DataRole.BENIGN_TEST:
            label = 0
        elif scores.role is DataRole.ATTACK_TEST:
            label = 1
        else:
            label = None
        return pd.DataFrame(
            {
                "dataset_id": identity.dataset.value,
                "client_id": scores.client_id.value,
                "row_id": [row_id.value for row_id in scores.row_ids],
                "phase": scores.role.value,
                "model_seed": int(identity.model_seed),
                "score_float64": np.asarray(scores.values, dtype=np.float64),
                "label_test_only": [label] * len(scores.values),
                "attack_family_test_only": (
                    [None if group is None else group.value for group in groups]
                    if scores.role is DataRole.ATTACK_TEST
                    else [None] * len(scores.values)
                ),
            }
        )

    @staticmethod
    def _validate_stream_roles(observed: dict[ClientId, set[DataRole]]) -> None:
        required = {
            DataRole.TRAIN,
            DataRole.RESERVOIR,
            DataRole.BENIGN_TEST,
            DataRole.ATTACK_DEV,
            DataRole.ATTACK_TEST,
        }
        if not observed:
            raise ValueError("Score cache has no clients")
        for client_id, roles in observed.items():
            if roles != required:
                missing = required - roles
                extra = roles - required
                raise ValueError(
                    f"Score-cache role contract failed for {client_id.value}; "
                    f"missing={sorted(role.value for role in missing)}, "
                    f"extra={sorted(role.value for role in extra)}"
                )

    @classmethod
    def _metadata_payload(
        cls,
        identity: ScoreCacheIdentity,
        cache_hash: Sha256,
        client_ids: tuple[ClientId, ...],
        role_hashes: dict[str, dict[str, str]],
        role_counts: dict[str, dict[str, int]],
    ) -> dict[str, object]:
        return {
            "dataset": identity.dataset,
            "model_seed": int(identity.model_seed),
            "model_hash": identity.model_hash,
            "data_spec_hash": identity.data_spec_hash,
            "training_spec_hash": identity.training_spec_hash,
            "dataset_manifest_hash": identity.dataset_manifest_hash,
            "preprocessing_hash": identity.preprocessing_hash,
            "score_cache_file": cls.filename,
            "score_cache_sha256": cache_hash,
            "client_ids": client_ids,
            "role_hashes": role_hashes,
            "role_counts": role_counts,
        }

    def load_descriptor(self, root: Path) -> ScoreCacheDescriptor:
        metadata = self._read_verified_metadata(root)
        identity = ScoreCacheIdentity(
            dataset=DatasetId(str(metadata["dataset"])),
            model_seed=ModelSeed(int(metadata["model_seed"])),
            model_hash=Sha256(str(metadata["model_hash"])),
            data_spec_hash=Sha256(str(metadata["data_spec_hash"])),
            training_spec_hash=Sha256(str(metadata["training_spec_hash"])),
            dataset_manifest_hash=Sha256(str(metadata["dataset_manifest_hash"])),
            preprocessing_hash=Sha256(str(metadata["preprocessing_hash"])),
        )
        return ScoreCacheDescriptor(
            identity=identity,
            cache_sha256=Sha256(str(metadata["score_cache_sha256"])),
            client_ids=tuple(ClientId(str(value)) for value in metadata["client_ids"]),
            role_hashes={
                str(client): {str(role): str(value) for role, value in roles.items()}
                for client, roles in metadata["role_hashes"].items()
            },
            role_counts={
                str(client): {str(role): int(value) for role, value in roles.items()}
                for client, roles in metadata["role_counts"].items()
            },
        )

    def load(self, root: Path) -> ScoreManifest:
        """Materialize the full score cache for deliberately small in-memory workflows."""

        metadata = self._read_verified_metadata(root)
        parquet_path = root / str(metadata["score_cache_file"])
        frame = pd.read_parquet(parquet_path, engine="pyarrow")
        clients: dict[ClientId, ClientScoreSet] = {}
        for client_id, client_frame in frame.groupby("client_id", sort=True):
            score_map: dict[DataRole, RoleScores] = {}
            for role_value, role_frame in client_frame.groupby("phase", sort=True):
                role = DataRole(str(role_value))
                role_scores = self._role_scores_from_frame(
                    ClientId(str(client_id)), role, role_frame
                )
                expected = metadata["role_hashes"][str(client_id)][role.value]
                if role_scores.sha256.value != expected:
                    raise ValueError(
                        f"SCORE_CACHE_HASH_MISMATCH: {client_id}/{role.value}"
                    )
                score_map[role] = role_scores
            typed_client_id = ClientId(str(client_id))
            clients[typed_client_id] = ClientScoreSet(typed_client_id, score_map)

        manifest = ScoreManifest(
            dataset=DatasetId(str(metadata["dataset"])),
            model_seed=ModelSeed(int(metadata["model_seed"])),
            model_hash=Sha256(str(metadata["model_hash"])),
            data_spec_hash=Sha256(str(metadata["data_spec_hash"])),
            training_spec_hash=Sha256(str(metadata["training_spec_hash"])),
            dataset_manifest_hash=Sha256(str(metadata["dataset_manifest_hash"])),
            preprocessing_hash=Sha256(str(metadata["preprocessing_hash"])),
            clients=clients,
            cache_sha256=Sha256(str(metadata["score_cache_sha256"])),
        )
        validate_score_manifest(manifest)
        return manifest

    def read_role(self, root: Path, client_id: ClientId, role: DataRole) -> RoleScores:
        """Load and hash-verify one client/role partition from the immutable cache."""

        metadata = self._read_verified_metadata(root)
        parquet_path = root / str(metadata["score_cache_file"])
        frame = pd.read_parquet(
            parquet_path,
            engine="pyarrow",
            filters=[("client_id", "==", client_id.value), ("phase", "==", role.value)],
        )
        if frame.empty:
            raise KeyError(f"Score cache has no {client_id.value}/{role.value} partition")
        scores = self._role_scores_from_frame(client_id, role, frame)
        expected = metadata["role_hashes"][client_id.value][role.value]
        if scores.sha256.value != expected:
            raise ValueError(f"SCORE_CACHE_HASH_MISMATCH: {client_id.value}/{role.value}")
        return scores

    def iter_clients(
        self,
        root: Path,
        roles: tuple[DataRole, ...],
    ) -> Iterable[ClientScoreSet]:
        descriptor = self.load_descriptor(root)
        for client_id in descriptor.client_ids:
            score_map = {role: self.read_role(root, client_id, role) for role in roles}
            yield ClientScoreSet(client_id, score_map)

    def _read_verified_metadata(self, root: Path) -> dict[str, object]:
        metadata_path = root / self.manifest_filename
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        parquet_path = root / str(metadata["score_cache_file"])
        expected_hash = Sha256(str(metadata["score_cache_sha256"]))
        actual_hash = Sha256(sha256_file(parquet_path))
        if actual_hash != expected_hash:
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: serialized cache hash differs")
        return metadata

    @staticmethod
    def _role_scores_from_frame(
        client_id: ClientId,
        role: DataRole,
        frame: pd.DataFrame,
    ) -> RoleScores:
        group_values = frame["attack_family_test_only"]
        groups = None
        if group_values.notna().any():
            groups = tuple(
                AttackGroupId(str(value))
                for value in group_values.dropna().astype(str)
            )
            if len(groups) != len(frame):
                raise ValueError(
                    f"Attack-group metadata is incomplete for {client_id.value}/{role.value}"
                )
        return RoleScores(
            role=role,
            values=frame["score_float64"].to_numpy(dtype=np.float64),
            client_id=client_id,
            row_ids=tuple(RowId(str(value)) for value in frame["row_id"].astype(str)),
            attack_groups=groups,
        )
