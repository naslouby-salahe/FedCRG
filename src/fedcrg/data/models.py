"""Typed natural-client data, base partitions, calibration assignments, and eligibility."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fedcrg.core.enums import (
    CalibrationAssignmentMode,
    ChronologyStatus,
    DataRole,
    DatasetId,
    EligibilityStatus,
    FailureCode,
)
from fedcrg.core.ids import CalibrationSeed, ClientId, Sha256


@dataclass(frozen=True, slots=True)
class ClientData:
    dataset: DatasetId
    client_id: ClientId
    benign: pd.DataFrame
    attack: pd.DataFrame
    chronology: ChronologyStatus = ChronologyStatus.SOURCE_ORDER_ONLY


@dataclass(frozen=True, slots=True)
class RoleFrame:
    role: DataRole
    frame: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ClientSplits:
    """A typed set of materialized role frames for one client."""

    client_id: ClientId
    roles: tuple[RoleFrame, ...]

    def get(self, role: DataRole) -> pd.DataFrame:
        for item in self.roles:
            if item.role is role:
                return item.frame
        raise KeyError(role.value)

    def try_get(self, role: DataRole) -> pd.DataFrame | None:
        for item in self.roles:
            if item.role is role:
                return item.frame
        return None


@dataclass(frozen=True, slots=True)
class RolePositions:
    role: DataRole
    positions: tuple[int, ...]
    row_id_hash: Sha256


@dataclass(frozen=True, slots=True)
class CalibrationRoleAssignment:
    """Positions within the frozen benign reservoir assigned to each policy role."""

    client_id: ClientId
    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    roles: tuple[RolePositions, ...]

    def positions_for(self, role: DataRole) -> tuple[int, ...]:
        if role not in {
            DataRole.REFERENCE,
            DataRole.MISMATCH,
            DataRole.CALIBRATION,
            DataRole.BENIGN_GUARD,
        }:
            raise ValueError(f"{role.value} is not a calibration-reservoir role")
        for item in self.roles:
            if item.role is role:
                return item.positions
        raise KeyError(role.value)

    def row_id_hash_for(self, role: DataRole) -> Sha256:
        for item in self.roles:
            if item.role is role:
                return item.row_id_hash
        raise KeyError(role.value)


@dataclass(frozen=True, slots=True)
class EligibilityRecord:
    client_id: ClientId
    status: EligibilityStatus
    benign_count: int
    malicious_count: int
    attack_development_capacity: int
    primary_code: FailureCode | None
    secondary_codes: tuple[FailureCode, ...]
    chronology: ChronologyStatus
