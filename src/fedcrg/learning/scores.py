"""Anomaly score computation, calibration-role views, and the immutable score cache."""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, model_validator

from fedcrg.config import DatasetConfig
from fedcrg.hashing import sha256_file
from fedcrg.learning.detectors import DetectorModel
from fedcrg.paths import ScoreCacheLayout
from fedcrg.runtime import resolve_compute_device
from fedcrg.types import (
    AttackGroupId,
    BatchSize,
    CalibrationAssignmentMode,
    CalibrationSeed,
    ClientId,
    ComputeDeviceId,
    DataRole,
    DatasetId,
    Identifier,
    ImmutableRunError,
    ModelSeed,
    PreparedColumn,
    PositiveCount,
    Position,
    RngSeed,
    RowId,
    SampleCount,
    Sha256,
)

Frozen = ConfigDict(frozen=True)


class ScoreCacheColumn(StrEnum):
    """Column names of the persisted score-cache parquet table."""

    DATASET_ID = "dataset_id"
    CLIENT_ID = "client_id"
    PHASE = "phase"
    MODEL_SEED = "model_seed"
    SCORE_FLOAT64 = "score_float64"
    LABEL_TEST_ONLY = "label_test_only"
    ATTACK_FAMILY_TEST_ONLY = "attack_family_test_only"


# Calibration roles are derived per-seed views over the reservoir role and must
# never be written into the physical score cache; only the base roles below are persisted.
_CALIBRATION_ROLES = frozenset(
    (
        DataRole.REFERENCE,
        DataRole.MISMATCH,
        DataRole.CALIBRATION,
        DataRole.BENIGN_GUARD,
    )
)

_BASE_SCORE_ROLES = frozenset(
    (
        DataRole.TRAIN,
        DataRole.RESERVOIR,
        DataRole.BENIGN_TEST,
        DataRole.ATTACK_DEV,
        DataRole.ATTACK_TEST,
    )
)

_REQUIRED_BASE_ROLES = _BASE_SCORE_ROLES
_FORBIDDEN_DERIVED_ROLES = _CALIBRATION_ROLES


class RoleScoreInput(BaseModel):
    """Feature values to be scored for one client/role pair, before scoring runs."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    role: DataRole
    values: np.ndarray
    row_ids: tuple[RowId, ...]
    attack_groups: tuple[AttackGroupId, ...] | None = None

    @model_validator(mode="after")
    def _validate(self) -> RoleScoreInput:
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise ValueError("Detector inputs must be a two-dimensional feature matrix")
        if len(values) != len(self.row_ids):
            raise ValueError("row_ids must align with detector inputs")
        if self.attack_groups is not None and len(values) != len(self.attack_groups):
            raise ValueError("attack_groups must align with detector inputs")
        return self


class ClientScoreInput(BaseModel):
    """Unscored feature inputs for one client, grouped by role."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    client_id: ClientId
    roles: tuple[RoleScoreInput, ...]

    def get(self, role: DataRole) -> RoleScoreInput:
        """Return this client's input for the given role."""
        for item in self.roles:
            if item.role is role:
                return item
        raise KeyError(role)


class RoleScores(BaseModel):
    """Computed anomaly scores for one client/role pair, with row provenance."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    role: DataRole
    values: np.ndarray
    client_id: ClientId
    row_ids: tuple[RowId, ...]
    attack_groups: tuple[AttackGroupId, ...] | None = None

    @model_validator(mode="after")
    def _validate(self) -> RoleScores:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("Role scores must be one-dimensional")
        if len(values) != len(self.row_ids):
            raise ValueError("Row provenance must align with score values")
        if self.attack_groups is not None and len(self.attack_groups) != len(values):
            raise ValueError("Attack-group metadata must align with score values")
        if not np.isfinite(values).all():
            raise ValueError("NONFINITE_SCORE: cached anomaly scores must be finite")
        return self

    @property
    def sha256(self) -> Sha256:
        """Hash of role, client id, row ids, values, and attack groups, used to detect tampering."""
        digest = hashlib.sha256()
        digest.update(self.role.encode("utf-8"))
        digest.update(self.client_id.encode("utf-8"))
        for row_id, value in zip(self.row_ids, self.values, strict=True):
            digest.update(row_id.encode("ascii"))
            digest.update(np.float64(value).tobytes())
        if self.attack_groups is not None:
            for group in self.attack_groups:
                digest.update(group.encode("utf-8"))
        return digest.hexdigest()


class RoleHash(BaseModel):
    """Score hash for a single role, without the underlying values."""

    model_config = Frozen

    role: DataRole
    sha256: Sha256


class ClientRoleHashes(BaseModel):
    """Score hashes for every role belonging to one client."""

    model_config = Frozen

    client_id: ClientId
    roles: tuple[RoleHash, ...]


class ClientScoreSet(BaseModel):
    """Computed scores for one client, grouped by role."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    client_id: ClientId
    scores: tuple[RoleScores, ...]

    def get(self, role: DataRole) -> RoleScores:
        """Return this client's scores for the given role."""
        for item in self.scores:
            if item.role is role:
                return item
        raise KeyError(role)


class ScoreManifest(BaseModel):
    """Every client's scores for a trained model, tied to the hashes that identify the model and data."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

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
        """Return the score set for the given client."""
        for item in self.clients:
            if item.client_id == client_id:
                return item
        raise KeyError(client_id)

    def role_hashes(self) -> tuple[ClientRoleHashes, ...]:
        """Return per-client, per-role score hashes without the underlying values."""
        return tuple(
            ClientRoleHashes(
                client_id=client.client_id,
                roles=tuple(
                    RoleHash(role=role_scores.role, sha256=role_scores.sha256)
                    for role_scores in client.scores
                ),
            )
            for client in self.clients
        )


class ClientCalibrationScores(BaseModel):
    """One client's reservoir scores split into calibration roles for a given calibration seed."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    client_id: ClientId
    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    roles: tuple[RoleScores, ...]

    def get(self, role: DataRole) -> RoleScores:
        """Return this client's scores for the given calibration role."""
        for item in self.roles:
            if item.role is role:
                return item
        raise KeyError(role)


class CalibrationScoreViews(BaseModel):
    """Calibration-role score views for every client under a given calibration seed."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    clients: tuple[ClientCalibrationScores, ...]

    @property
    def client_ids(self) -> tuple[ClientId, ...]:
        """Sorted ids of clients covered by this view."""
        return tuple(sorted(item.client_id for item in self.clients))

    def client(self, client_id: ClientId) -> ClientCalibrationScores:
        """Return the calibration scores for the given client."""
        for item in self.clients:
            if item.client_id == client_id:
                return item
        raise KeyError(client_id)

    def get(self, client_id: ClientId, role: DataRole) -> RoleScores:
        """Return one client's scores for the given calibration role."""
        return self.client(client_id).get(role)


def truncate_view(role_scores: RoleScores, sample_count: SampleCount) -> RoleScores:
    """Return the first `sample_count` rows of a role's scores, dropping attack-group metadata."""
    if sample_count <= 0 or sample_count > len(role_scores.values):
        raise ValueError(
            f"Cannot take {sample_count} values from {len(role_scores.values)} {role_scores.role} scores"
        )
    return RoleScores(
        role=role_scores.role,
        values=role_scores.values[:sample_count],
        client_id=role_scores.client_id,
        row_ids=role_scores.row_ids[:sample_count],
        attack_groups=None,
    )


class CalibrationAssignment:
    """Partitions a client's reservoir rows into calibration roles by seeded permutation, so each calibration_seed yields a reproducible but distinct split."""

    def __init__(
        self,
        client_id: ClientId,
        dataset: DatasetConfig,
        calibration_seed: CalibrationSeed,
        mode: CalibrationAssignmentMode,
        reservoir_row_ids: tuple[RowId, ...],
    ) -> None:
        split = dataset.split
        total_len = (
            split.reference_benign
            + split.mismatch_benign
            + split.calibration_benign
            + split.benign_guard
        )
        positions = np.arange(total_len)

        if mode is CalibrationAssignmentMode.SEEDED_PERMUTATION:
            rng = np.random.Generator(
                np.random.PCG64(_assignment_seed(dataset.id, calibration_seed, client_id))
            )
            positions = rng.permutation(positions)

        self._positions: dict[DataRole, tuple[int, ...]] = {
            DataRole.REFERENCE: tuple(positions[: split.reference_benign].tolist()),
            DataRole.MISMATCH: tuple(
                positions[
                    split.reference_benign : split.reference_benign + split.mismatch_benign
                ].tolist()
            ),
            DataRole.CALIBRATION: tuple(
                positions[
                    split.reference_benign + split.mismatch_benign : split.reference_benign
                    + split.mismatch_benign
                    + split.calibration_benign
                ].tolist()
            ),
            DataRole.BENIGN_GUARD: tuple(
                positions[
                    split.reference_benign + split.mismatch_benign + split.calibration_benign :
                ].tolist()
            ),
        }
        self._row_ids = reservoir_row_ids

    def positions_for(self, role: DataRole) -> tuple[Position, ...]:
        """Reservoir row positions assigned to the given calibration role."""
        return self._positions[role]

    def row_id_hash_for(self, role: DataRole) -> Sha256:
        """Hash of the row ids assigned to the given calibration role, in assignment order."""
        digest = hashlib.sha256()
        for index in self._positions[role]:
            digest.update(self._row_ids[index].encode("ascii"))
        return digest.hexdigest()


def _assignment_seed(
    dataset_id: DatasetId, calibration_seed: CalibrationSeed, client_id: ClientId
) -> RngSeed:
    text = f"fedcrg|{dataset_id}|calibration|{calibration_seed}|{client_id}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class CalibrationScoreViewBuilder:
    """Builds per-client calibration-role score views from a score manifest's reservoir scores."""

    def build(
        self,
        scores: ScoreManifest,
        dataset: DatasetConfig,
        calibration_seed: CalibrationSeed,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> CalibrationScoreViews:
        """Split each client's reservoir scores into calibration roles for the given seed."""
        if scores.dataset is not dataset.id:
            raise ValueError("Score cache dataset does not match calibration configuration")
        reservoirs = {
            client_scores.client_id: client_scores.get(DataRole.RESERVOIR)
            for client_scores in scores.clients
        }
        result: list[ClientCalibrationScores] = []
        for client_id, reservoir in sorted(reservoirs.items()):
            assignment = CalibrationAssignment(
                client_id=client_id,
                dataset=dataset,
                calibration_seed=calibration_seed,
                mode=mode,
                reservoir_row_ids=reservoir.row_ids,
            )
            role_views: list[RoleScores] = []
            for role in _CALIBRATION_ROLES:
                positions = assignment.positions_for(role)
                role_views.append(
                    RoleScores(
                        role=role,
                        values=reservoir.values[list(positions)],
                        client_id=client_id,
                        row_ids=tuple(reservoir.row_ids[index] for index in positions),
                    )
                )
            result.append(
                ClientCalibrationScores(
                    client_id=client_id,
                    calibration_seed=calibration_seed,
                    mode=mode,
                    roles=tuple(role_views),
                )
            )
        return CalibrationScoreViews(
            calibration_seed=calibration_seed, mode=mode, clients=tuple(result)
        )


class ScoreCacheMetadataRecord(BaseModel):
    """Hash and row count for one client/role score block, for tamper detection without reading the full cache."""

    model_config = Frozen

    client_id: ClientId
    role: DataRole
    score_array_sha256: Sha256
    row_count: PositiveCount


class ScoreCacheMetadata(BaseModel):
    """On-disk sidecar describing a score cache's identity, file, and hash."""

    model_config = Frozen

    dataset: DatasetId
    model_seed: ModelSeed
    model_hash: Sha256
    data_spec_hash: Sha256
    training_spec_hash: Sha256
    dataset_manifest_hash: Sha256
    preprocessing_hash: Sha256
    score_cache_file: Identifier
    score_cache_sha256: Sha256
    client_ids: tuple[ClientId, ...]
    records: tuple[ScoreCacheMetadataRecord, ...]


@dataclass(frozen=True, slots=True)
class ScoreCacheIdentity:
    """Hashes that pin a score cache to the exact model and data it was computed from."""

    dataset: DatasetId
    model_seed: ModelSeed
    model_hash: Sha256
    data_spec_hash: Sha256
    training_spec_hash: Sha256
    dataset_manifest_hash: Sha256
    preprocessing_hash: Sha256


@dataclass(frozen=True, slots=True)
class ScoreRoleCacheRecord:
    """In-memory counterpart of `ScoreCacheMetadataRecord`."""

    client_id: ClientId
    role: DataRole
    score_array_sha256: Sha256
    row_count: PositiveCount


@dataclass(frozen=True, slots=True)
class ScoreCacheDescriptor:
    """Identity and record hashes for a written or loaded score cache, without the score values themselves."""

    identity: ScoreCacheIdentity
    cache_sha256: Sha256
    client_ids: tuple[ClientId, ...]
    records: tuple[ScoreRoleCacheRecord, ...]


class ScoreCache:
    """Reads and writes score caches on disk; once written, a cache directory is never modified in place."""

    def save(self, manifest: ScoreManifest, root: Path) -> ScoreManifest:
        """Validate and persist a score manifest as an immutable cache directory."""
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
                for client in sorted(manifest.clients, key=lambda item: item.client_id)
                for role_scores in client.scores
            ),
            root,
        )
        return ScoreManifest.model_validate(
            {**manifest.model_dump(mode="python"), "cache_sha256": descriptor.cache_sha256}
        )

    def save_stream(
        self,
        identity: ScoreCacheIdentity,
        role_scores: Iterable[RoleScores],
        root: Path,
    ) -> ScoreCacheDescriptor:
        """Write role scores to a staging directory and atomically publish it as the cache root."""
        if root.exists():
            raise ImmutableRunError(f"Score cache already exists and is immutable: {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        staging = root.parent / f".{root.name}.staging-{uuid.uuid4().hex}"
        staging.mkdir()
        parquet_path = ScoreCacheLayout(staging).score
        writer = None
        records: list[ScoreRoleCacheRecord] = []
        observed_roles: dict[ClientId, set[DataRole]] = {}

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            for scores in role_scores:
                if scores.role in _CALIBRATION_ROLES:
                    raise ValueError(
                        "Calibration-assignment roles are views and must not be serialized in the physical score cache"
                    )
                cid = scores.client_id
                observed_roles.setdefault(cid, set())
                if scores.role in observed_roles[cid]:
                    raise ValueError(f"Duplicate score block: {cid}/{scores.role}")
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
                records.append(
                    ScoreRoleCacheRecord(
                        client_id=cid,
                        role=scores.role,
                        score_array_sha256=scores.sha256,
                        row_count=len(scores.values),
                    )
                )

            if writer is None:
                raise ValueError("Cannot finalize an empty score cache")
            writer.close()
            writer = None
            self._validate_stream_roles(observed_roles)
            cache_hash = sha256_file(parquet_path)
            client_ids = tuple(sorted(observed_roles))
            frozen_records = tuple(sorted(records, key=lambda item: (item.client_id, item.role)))
            self._write_metadata(staging, identity, cache_hash, client_ids, frozen_records)
            os.replace(staging, root)
            return ScoreCacheDescriptor(
                identity=identity,
                cache_sha256=cache_hash,
                client_ids=client_ids,
                records=frozen_records,
            )
        except Exception:
            if writer is not None:
                writer.close()
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _role_frame(identity: ScoreCacheIdentity, scores: RoleScores) -> pd.DataFrame:
        groups = scores.attack_groups
        n_samples = len(scores.values)

        if scores.role is DataRole.BENIGN_TEST:
            label: int | None = 0
        elif scores.role is DataRole.ATTACK_TEST:
            label = 1
        else:
            label = None

        return pd.DataFrame(
            {
                ScoreCacheColumn.DATASET_ID: identity.dataset,
                ScoreCacheColumn.CLIENT_ID: scores.client_id,
                PreparedColumn.ROW_ID: list(scores.row_ids),
                ScoreCacheColumn.PHASE: scores.role,
                ScoreCacheColumn.MODEL_SEED: int(identity.model_seed),
                ScoreCacheColumn.SCORE_FLOAT64: np.asarray(scores.values, dtype=np.float64),
                ScoreCacheColumn.LABEL_TEST_ONLY: pd.array([label] * n_samples, dtype="Int64"),
                ScoreCacheColumn.ATTACK_FAMILY_TEST_ONLY: pd.array(
                    (
                        list(groups)
                        if scores.role is DataRole.ATTACK_TEST and groups is not None
                        else [None] * n_samples
                    ),
                    dtype="string",
                ),
            }
        )

    @staticmethod
    def _validate_stream_roles(observed: dict[ClientId, set[DataRole]]) -> None:
        if not observed:
            raise ValueError("Score cache has no clients")
        for client_id, roles in observed.items():
            if roles != set(_REQUIRED_BASE_ROLES):
                missing = _REQUIRED_BASE_ROLES - roles
                extra = roles - _REQUIRED_BASE_ROLES
                raise ValueError(
                    f"Score-cache role contract failed for {client_id}, "
                    f"missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )

    @classmethod
    def _write_metadata(
        cls,
        staging: Path,
        identity: ScoreCacheIdentity,
        cache_hash: Sha256,
        client_ids: tuple[ClientId, ...],
        records: tuple[ScoreRoleCacheRecord, ...],
    ) -> None:
        payload = ScoreCacheMetadata(
            dataset=identity.dataset,
            model_seed=identity.model_seed,
            model_hash=identity.model_hash,
            data_spec_hash=identity.data_spec_hash,
            training_spec_hash=identity.training_spec_hash,
            dataset_manifest_hash=identity.dataset_manifest_hash,
            preprocessing_hash=identity.preprocessing_hash,
            score_cache_file=ScoreCacheLayout(staging).score.name,
            score_cache_sha256=cache_hash,
            client_ids=client_ids,
            records=tuple(
                ScoreCacheMetadataRecord(
                    client_id=record.client_id,
                    role=record.role,
                    score_array_sha256=record.score_array_sha256,
                    row_count=record.row_count,
                )
                for record in records
            ),
        )
        ScoreCacheLayout(staging).manifest.write_text(
            payload.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    def load_descriptor(self, root: Path) -> ScoreCacheDescriptor:
        """Read cache identity and record hashes without loading the score values."""
        metadata = self._read_verified_metadata(root)
        identity = ScoreCacheIdentity(
            dataset=metadata.dataset,
            model_seed=metadata.model_seed,
            model_hash=metadata.model_hash,
            data_spec_hash=metadata.data_spec_hash,
            training_spec_hash=metadata.training_spec_hash,
            dataset_manifest_hash=metadata.dataset_manifest_hash,
            preprocessing_hash=metadata.preprocessing_hash,
        )
        return ScoreCacheDescriptor(
            identity=identity,
            cache_sha256=metadata.score_cache_sha256,
            client_ids=metadata.client_ids,
            records=self._records_from_metadata(metadata),
        )

    @classmethod
    def _records_from_metadata(
        cls, metadata: ScoreCacheMetadata
    ) -> tuple[ScoreRoleCacheRecord, ...]:
        return tuple(
            ScoreRoleCacheRecord(
                client_id=record.client_id,
                role=record.role,
                score_array_sha256=record.score_array_sha256,
                row_count=record.row_count,
            )
            for record in metadata.records
        )

    def load(self, root: Path) -> ScoreManifest:
        """Load and hash-verify the full score manifest from a cache directory."""
        metadata = self._read_verified_metadata(root)
        records = self._records_from_metadata(metadata)
        parquet_path = root / metadata.score_cache_file
        frame = pd.read_parquet(parquet_path)
        clients: list[ClientScoreSet] = []
        for client_id, client_frame in frame.groupby(ScoreCacheColumn.CLIENT_ID, sort=True):
            score_list: list[RoleScores] = []
            typed_client_id = str(client_id)
            for role_value, role_frame in client_frame.groupby(ScoreCacheColumn.PHASE, sort=True):
                role = DataRole(str(role_value))
                role_scores = self._role_scores_from_frame(typed_client_id, role, role_frame)
                expected = self._record_lookup(records, typed_client_id, role).score_array_sha256
                if role_scores.sha256 != expected:
                    raise ValueError(f"SCORE_CACHE_HASH_MISMATCH: {client_id}/{role}")
                score_list.append(role_scores)
            clients.append(ClientScoreSet(client_id=typed_client_id, scores=tuple(score_list)))

        manifest = ScoreManifest(
            dataset=metadata.dataset,
            model_seed=metadata.model_seed,
            model_hash=metadata.model_hash,
            data_spec_hash=metadata.data_spec_hash,
            training_spec_hash=metadata.training_spec_hash,
            dataset_manifest_hash=metadata.dataset_manifest_hash,
            preprocessing_hash=metadata.preprocessing_hash,
            clients=tuple(clients),
            cache_sha256=metadata.score_cache_sha256,
        )
        validate_score_manifest(manifest)
        return manifest

    def read_role(self, root: Path, client_id: ClientId, role: DataRole) -> RoleScores:
        """Load and hash-verify one client/role score block without reading the rest of the cache."""
        metadata = self._read_verified_metadata(root)
        parquet_path = root / metadata.score_cache_file
        frame = pd.read_parquet(
            parquet_path,
            filters=[
                (ScoreCacheColumn.CLIENT_ID, "==", client_id),
                (ScoreCacheColumn.PHASE, "==", role),
            ],
        )
        if frame.empty:
            raise KeyError(f"Score cache has no {client_id}/{role} partition")
        scores = self._role_scores_from_frame(client_id, role, frame)
        records = self._records_from_metadata(metadata)
        expected = self._record_lookup(records, client_id, role).score_array_sha256
        if scores.sha256 != expected:
            raise ValueError(f"SCORE_CACHE_HASH_MISMATCH: {client_id}/{role}")
        return scores

    def _read_verified_metadata(self, root: Path) -> ScoreCacheMetadata:
        metadata_path = ScoreCacheLayout(root).manifest
        metadata = ScoreCacheMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        parquet_path = root / metadata.score_cache_file
        actual_hash = sha256_file(parquet_path)
        if actual_hash != metadata.score_cache_sha256:
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: serialized cache hash differs")
        return metadata

    @staticmethod
    def _record_lookup(
        records: tuple[ScoreRoleCacheRecord, ...],
        client_id: ClientId,
        role: DataRole,
    ) -> ScoreRoleCacheRecord:
        for record in records:
            if record.client_id == client_id and record.role is role:
                return record
        raise KeyError(f"{client_id}/{role}")

    @staticmethod
    def _role_scores_from_frame(
        client_id: ClientId,
        role: DataRole,
        frame: pd.DataFrame,
    ) -> RoleScores:
        group_values = frame[ScoreCacheColumn.ATTACK_FAMILY_TEST_ONLY]
        groups = None
        if bool(group_values.notna().any()):
            groups = tuple(str(value) for value in group_values.dropna().astype(str))
            if len(groups) != len(frame):
                raise ValueError(f"Attack-group metadata is incomplete for {client_id}/{role}")
        return RoleScores(
            role=role,
            values=frame[ScoreCacheColumn.SCORE_FLOAT64].to_numpy(dtype=np.float64),
            client_id=client_id,
            row_ids=tuple(str(value) for value in frame[PreparedColumn.ROW_ID].astype(str)),
            attack_groups=groups,
        )


def _validate_client_role_coverage(client: ClientScoreSet) -> None:
    client_id = client.client_id
    present = {item.role for item in client.scores}
    missing = _REQUIRED_BASE_ROLES - present
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Score manifest {client_id} is missing base roles: {names}")
    duplicated = _FORBIDDEN_DERIVED_ROLES & present
    if duplicated:
        names = ", ".join(sorted(duplicated))
        raise ValueError(f"Score cache must not materialize calibration-assignment roles: {names}")


def _validate_client_role_scores(client: ClientScoreSet) -> None:
    client_id = client.client_id
    row_ids: set[RowId] = set()
    for role_scores in client.scores:
        role = role_scores.role
        if role_scores.values.dtype != np.float64:
            raise ValueError("Cached scores must be float64")
        if not np.isfinite(role_scores.values).all():
            raise ValueError("NONFINITE_SCORE")
        overlap = row_ids.intersection(role_scores.row_ids)
        if overlap:
            raise ValueError(f"ROLE_OVERLAP: {client_id}/{role}")
        row_ids.update(role_scores.row_ids)
        if len(role_scores.sha256) != 64:
            raise ValueError("Invalid role-score hash")
    if len(client.get(DataRole.BENIGN_TEST).values) == 0:
        raise ValueError(f"Final benign test is empty for {client_id}")
    if len(client.get(DataRole.ATTACK_TEST).values) == 0:
        raise ValueError(f"Final attack test is empty for {client_id}")


def validate_score_manifest(manifest: ScoreManifest) -> None:
    """Check that every client has the required roles, finite float64 scores, and non-overlapping row ids."""
    if not manifest.clients:
        raise ValueError("Score manifest has no clients")
    for client in manifest.clients:
        _validate_client_role_coverage(client)
        _validate_client_role_scores(client)


class ScoreComputer:
    """Scores are label-blind: only feature values are read, never attack/malicious labels."""

    def compute(
        self,
        model: DetectorModel,
        values: np.ndarray,
        device: ComputeDeviceId,
        batch_size: BatchSize,
    ) -> np.ndarray:
        """Score every row of `values` in batches, returning float64 anomaly scores."""
        if batch_size <= 0:
            raise ValueError("Scoring batch_size must be positive")
        torch_device = resolve_compute_device(device)
        model = model.to(torch_device).eval()
        n_samples = len(values)
        if n_samples == 0:
            return np.empty(0, dtype=np.float64)

        out = np.empty(n_samples, dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                # Forward pass runs in float32; results are widened to float64 for storage
                # so downstream statistical calculations get consistent precision.
                batch = torch.as_tensor(
                    values[start:end],
                    dtype=torch.float32,
                    device=torch_device,
                )
                out[start:end] = (
                    model.anomaly_score(batch).detach().cpu().numpy().astype(np.float64, copy=False)
                )
        return out

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
        device: ComputeDeviceId,
        batch_size: BatchSize,
    ) -> ScoreManifest:
        """Score every role of every client and assemble the resulting manifest."""
        scored_clients: list[ClientScoreSet] = []
        for client in clients:
            role_scores = tuple(
                RoleScores(
                    role=role_input.role,
                    values=self.compute(model, role_input.values, device, batch_size),
                    client_id=client.client_id,
                    row_ids=role_input.row_ids,
                    attack_groups=role_input.attack_groups,
                )
                for role_input in client.roles
            )
            scored_clients.append(ClientScoreSet(client_id=client.client_id, scores=role_scores))
        return ScoreManifest(
            dataset=dataset,
            model_seed=model_seed,
            model_hash=model.state_hash(),
            data_spec_hash=data_spec_hash,
            training_spec_hash=training_spec_hash,
            dataset_manifest_hash=dataset_manifest_hash,
            preprocessing_hash=preprocessing_hash,
            clients=tuple(scored_clients),
        )
