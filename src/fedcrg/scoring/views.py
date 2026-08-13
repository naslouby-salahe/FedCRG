"""Deterministic calibration-role views over one frozen reservoir score cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fedcrg.config.models import DatasetConfig
from fedcrg.core.enums import CalibrationAssignmentMode, DataRole
from fedcrg.core.ids import ClientId
from fedcrg.data.splitting import CalibrationAssignmentBuilder
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
    calibration_seed: int
    mode: CalibrationAssignmentMode
    roles: dict[DataRole, RoleScores]

    def get(self, role: DataRole) -> RoleScores:
        return self.roles[role]


@dataclass(frozen=True, slots=True)
class CalibrationScoreViews:
    calibration_seed: int
    mode: CalibrationAssignmentMode
    clients: dict[ClientId, ClientCalibrationScores]

    def get(self, client_id: ClientId, role: DataRole) -> RoleScores:
        return self.clients[client_id].get(role)


class CalibrationScoreViewBuilder:
    def __init__(self, assignments: CalibrationAssignmentBuilder | None = None) -> None:
        self.assignments = assignments or CalibrationAssignmentBuilder()

    def build(
        self,
        scores: ScoreManifest,
        dataset: DatasetConfig,
        calibration_seed: int,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
        prepared_root: Path | None = None,
    ) -> CalibrationScoreViews:
        if scores.dataset is not dataset.id:
            raise ValueError("Score cache dataset does not match calibration configuration")

        expected = self._load_expected_manifest(prepared_root, calibration_seed, mode)
        result: dict[ClientId, ClientCalibrationScores] = {}
        for client_id, client_scores in sorted(scores.clients.items()):
            reservoir = client_scores.scores[DataRole.RESERVOIR]
            frame = pd.DataFrame({"row_id": reservoir.row_ids})
            assignment = self.assignments.build(
                frame,
                scores.dataset,
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
        calibration_seed: int,
        mode: CalibrationAssignmentMode,
    ) -> dict[str, object] | None:
        if prepared_root is None:
            return None
        path = (
            prepared_root / "splits" / "seeded" / f"c{calibration_seed}.json"
            if mode is CalibrationAssignmentMode.SEEDED_PERMUTATION
            else prepared_root / "splits" / "source_order.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen calibration-assignment manifest: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload["calibration_seed"]) != calibration_seed:
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
        if not isinstance(roles, dict) or not isinstance(roles.get(role.value), dict):
            raise ValueError(f"Assignment manifest is missing {client_id.value}/{role.value}")
        record = roles[role.value]
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
