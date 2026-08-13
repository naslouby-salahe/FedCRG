"""Deterministic calibration-role views over one frozen reservoir score cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fedcrg.config.models import DatasetConfig
from fedcrg.core.enums import CalibrationAssignmentMode, DataRole, DatasetId
from fedcrg.core.ids import CalibrationSeed, ClientId
from fedcrg.data.splitting import CalibrationAssignmentBuilder
from fedcrg.scoring.cache import ScoreCache
from fedcrg.scoring.models import RoleScores, ScoreManifest

_CALIBRATION_ROLES = (
    DataRole.REFERENCE,
    DataRole.MISMATCH,
    DataRole.CALIBRATION,
    DataRole.BENIGN_GUARD,
)


@dataclass(frozen=True, slots=True)
class ClientCalibrationScores:
    client_id: ClientId
    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    roles: dict[DataRole, RoleScores]

    def get(self, role: DataRole) -> RoleScores:
        return self.roles[role]


@dataclass(frozen=True, slots=True)
class CalibrationScoreViews:
    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    clients: dict[ClientId, ClientCalibrationScores]

    @property
    def client_ids(self) -> tuple[ClientId, ...]:
        return tuple(sorted(self.clients))

    def get(self, client_id: ClientId, role: DataRole) -> RoleScores:
        return self.clients[client_id].get(role)


class CalibrationScoreViewBuilder:
    def __init__(self, assignments: CalibrationAssignmentBuilder | None = None) -> None:
        self.assignments = assignments or CalibrationAssignmentBuilder()

    def build_from_cache(
        self,
        cache: ScoreCache,
        score_root: Path,
        dataset: DatasetConfig,
        calibration_seed: CalibrationSeed | int,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
        prepared_root: Path | None = None,
    ) -> CalibrationScoreViews:
        """Open reservoir scores only; final-test roles remain unopened."""

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
        calibration_seed: CalibrationSeed | int,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
        prepared_root: Path | None = None,
    ) -> CalibrationScoreViews:
        """Build from an intentionally materialized small score manifest."""

        if scores.dataset is not dataset.id:
            raise ValueError("Score cache dataset does not match calibration configuration")
        reservoirs = {
            client_id: client_scores.scores[DataRole.RESERVOIR]
            for client_id, client_scores in scores.clients.items()
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
        result: dict[ClientId, ClientCalibrationScores] = {}
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
            role_views: dict[DataRole, RoleScores] = {}
            for role in _CALIBRATION_ROLES:
                positions = assignment.positions_for(role)
                selected_values = reservoir.values[list(positions)]
                selected_ids = tuple(reservoir.row_ids[index] for index in positions)
                role_views[role] = RoleScores(
                    role=role,
                    values=selected_values,
                    client_id=client_id,
                    row_ids=selected_ids,
                )
                self._verify_expected(
                    expected,
                    client_id,
                    role,
                    len(selected_ids),
                    assignment.row_id_hashes[role].value,
                )
            result[client_id] = ClientCalibrationScores(
                client_id=client_id,
                calibration_seed=calibration_seed,
                mode=mode,
                roles=role_views,
            )
        return CalibrationScoreViews(calibration_seed, mode, result)

    @staticmethod
    def _load_expected_manifest(
        prepared_root: Path | None,
        calibration_seed: CalibrationSeed,
        mode: CalibrationAssignmentMode,
    ) -> dict[str, object] | None:
        if prepared_root is None:
            return None
        path = (
            prepared_root / "splits" / "seeded" / f"c{int(calibration_seed)}.json"
            if mode is CalibrationAssignmentMode.SEEDED_PERMUTATION
            else prepared_root / "splits" / "source_order.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen calibration-assignment manifest: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload["calibration_seed"]) != int(calibration_seed):
            raise ValueError("Calibration-assignment manifest seed mismatch")
        if str(payload["mode"]) != mode.value:
            raise ValueError("Calibration-assignment manifest mode mismatch")
        return payload

    @staticmethod
    def _verify_expected(
        expected: dict[str, object] | None,
        client_id: ClientId,
        role: DataRole,
        row_count: int,
        row_hash: str,
    ) -> None:
        if expected is None:
            return
        clients = expected.get("clients")
        if not isinstance(clients, list):
            raise ValueError("Malformed calibration-assignment manifest")
        client = next(
            (
                item
                for item in clients
                if isinstance(item, dict) and item.get("client_id") == client_id.value
            ),
            None,
        )
        if not isinstance(client, dict):
            raise ValueError(f"Assignment manifest is missing {client_id.value}")
        roles = client.get("roles")
        if not isinstance(roles, list):
            raise ValueError(f"Assignment manifest roles are malformed for {client_id.value}")
        record = next(
            (
                item
                for item in roles
                if isinstance(item, dict) and item.get("role") == role.value
            ),
            None,
        )
        if not isinstance(record, dict):
            raise ValueError(f"Assignment manifest is missing {client_id.value}/{role.value}")
        if int(record["row_count"]) != row_count or str(record["row_id_sha256"]) != row_hash:
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
