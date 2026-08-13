"""Typed anomaly-score inputs and deterministic calibration-role views."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from fedcrg.artifacts.manifests import CalibrationAssignmentManifestStore
from fedcrg.configuration.dataset_config import DatasetConfig
from fedcrg.datasets.splits import CalibrationAssignmentBuilder, CalibrationAssignmentManifest
from fedcrg.domain.enums import CalibrationAssignmentMode, DataRole, DatasetId
from fedcrg.domain.identifiers import AttackGroupId, CalibrationSeed, ClientId, RowId, Sha256

if TYPE_CHECKING:
    from fedcrg.scoring.cache import ScoreCache
    from fedcrg.scoring.score_records import ScoreManifest

_CALIBRATION_ROLES = (
    DataRole.REFERENCE,
    DataRole.MISMATCH,
    DataRole.CALIBRATION,
    DataRole.BENIGN_GUARD,
)


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
class ClientCalibrationScores:
    client_id: ClientId
    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    roles: tuple[RoleScores, ...]

    def get(self, role: DataRole) -> RoleScores:
        for item in self.roles:
            if item.role is role:
                return item
        raise KeyError(role.value)


@dataclass(frozen=True, slots=True)
class CalibrationScoreViews:
    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    clients: tuple[ClientCalibrationScores, ...]

    @property
    def client_ids(self) -> tuple[ClientId, ...]:
        return tuple(sorted(item.client_id for item in self.clients))

    def client(self, client_id: ClientId) -> ClientCalibrationScores:
        for item in self.clients:
            if item.client_id == client_id:
                return item
        raise KeyError(client_id.value)

    def get(self, client_id: ClientId, role: DataRole) -> RoleScores:
        return self.client(client_id).get(role)


class CalibrationScoreViewBuilder:
    def __init__(
        self,
        assignments: CalibrationAssignmentBuilder | None = None,
        assignment_manifests: CalibrationAssignmentManifestStore | None = None,
    ) -> None:
        self.assignments = assignments or CalibrationAssignmentBuilder()
        self.assignment_manifests = assignment_manifests or CalibrationAssignmentManifestStore()

    def build_from_cache(
        self,
        cache: ScoreCache,
        score_root: Path,
        dataset: DatasetConfig,
        calibration_seed: CalibrationSeed,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
        prepared_root: Path | None = None,
    ) -> CalibrationScoreViews:
        """Open reservoir scores only, final-test roles remain unopened."""

        seed = CalibrationSeed(int(calibration_seed))
        descriptor = cache.load_descriptor(score_root)
        if descriptor.identity.dataset is not dataset.id:
            raise ValueError("Score cache dataset does not match calibration configuration")
        reservoirs = {
            client_id: cache.read_role(score_root, client_id, DataRole.RESERVOIR)
            for client_id in descriptor.client_ids
        }
        return self._build_from_reservoirs(
            descriptor.identity.dataset,
            reservoirs,
            dataset,
            seed,
            mode,
            prepared_root,
        )

    def build(
        self,
        scores: ScoreManifest,
        dataset: DatasetConfig,
        calibration_seed: CalibrationSeed,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
        prepared_root: Path | None = None,
    ) -> CalibrationScoreViews:
        """Build from an intentionally materialized small score manifest."""

        if scores.dataset is not dataset.id:
            raise ValueError("Score cache dataset does not match calibration configuration")
        reservoirs = {
            client_scores.client_id: client_scores.get(DataRole.RESERVOIR)
            for client_scores in scores.clients
        }
        return self._build_from_reservoirs(
            scores.dataset,
            reservoirs,
            dataset,
            CalibrationSeed(int(calibration_seed)),
            mode,
            prepared_root,
        )

    def _build_from_reservoirs(
        self,
        dataset_id: DatasetId,
        reservoirs: dict[ClientId, RoleScores],
        dataset: DatasetConfig,
        calibration_seed: CalibrationSeed,
        mode: CalibrationAssignmentMode,
        prepared_root: Path | None,
    ) -> CalibrationScoreViews:
        expected = self._load_expected_manifest(prepared_root, calibration_seed, mode)
        result: list[ClientCalibrationScores] = []
        for client_id, reservoir in sorted(reservoirs.items()):
            frame = pd.DataFrame({"row_id": [row_id.value for row_id in reservoir.row_ids]})
            assignment = self.assignments.build(
                frame,
                dataset_id,
                client_id,
                dataset,
                calibration_seed,
                mode,
            )
            role_views: list[RoleScores] = []
            for role in _CALIBRATION_ROLES:
                positions = assignment.positions_for(role)
                selected_values = reservoir.values[list(positions)]
                selected_ids = tuple(reservoir.row_ids[index] for index in positions)
                role_views.append(
                    RoleScores(
                        role=role,
                        values=selected_values,
                        client_id=client_id,
                        row_ids=selected_ids,
                    )
                )
                self._verify_expected(
                    expected,
                    client_id,
                    role,
                    len(selected_ids),
                    assignment.row_id_hash_for(role).value,
                )
            result.append(
                ClientCalibrationScores(
                    client_id=client_id,
                    calibration_seed=calibration_seed,
                    mode=mode,
                    roles=tuple(role_views),
                )
            )
        return CalibrationScoreViews(calibration_seed, mode, tuple(result))

    def _load_expected_manifest(
        self,
        prepared_root: Path | None,
        calibration_seed: CalibrationSeed,
        mode: CalibrationAssignmentMode,
    ) -> CalibrationAssignmentManifest | None:
        if prepared_root is None:
            return None
        path = (
            prepared_root / "splits" / "seeded" / f"c{int(calibration_seed)}.json"
            if mode is CalibrationAssignmentMode.SEEDED_PERMUTATION
            else prepared_root / "splits" / "source_order.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen calibration-assignment manifest: {path}")
        manifest = self.assignment_manifests.load(path)
        if manifest.calibration_seed != calibration_seed:
            raise ValueError("Calibration-assignment manifest seed mismatch")
        if manifest.mode is not mode:
            raise ValueError("Calibration-assignment manifest mode mismatch")
        return manifest

    @staticmethod
    def _verify_expected(
        expected: CalibrationAssignmentManifest | None,
        client_id: ClientId,
        role: DataRole,
        row_count: int,
        row_hash: str,
    ) -> None:
        if expected is None:
            return
        record = expected.client(client_id).role(role)
        if record.row_count != row_count or record.row_id_sha256.value != row_hash:
            raise ValueError(
                f"Calibration assignment hash mismatch for {client_id.value}/{role.value}"
            )


def truncate_view(role_scores: RoleScores, sample_count: int) -> RoleScores:
    if sample_count <= 0 or sample_count > len(role_scores.values):
        raise ValueError(
            f"Cannot take {sample_count} values from {len(role_scores.values)} {role_scores.role.value} scores"
        )
    return RoleScores(
        role=role_scores.role,
        values=role_scores.values[:sample_count],
        client_id=role_scores.client_id,
        row_ids=role_scores.row_ids[:sample_count],
        attack_groups=None,
    )
