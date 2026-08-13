"""Typed dataset, eligibility, split, and prepared-cache provenance models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable

from fedcrg.domain.enums import (
    CalibrationAssignmentMode,
    DataRole,
    DatasetId,
    FailureCode,
)
from fedcrg.domain.identifiers import CalibrationSeed, ClientId, RowId, Sha256
from fedcrg.data.models import EligibilityRecord


@dataclass(frozen=True, slots=True)
class SourceFileManifest:
    relative_path: PurePosixPath
    sha256: Sha256
    size_bytes: int


@dataclass(frozen=True, slots=True)
class EligibilityManifest:
    dataset_id: DatasetId
    discovered_clients: tuple[ClientId, ...]
    eligible_clients: tuple[ClientId, ...]
    records: tuple[EligibilityRecord, ...]


@dataclass(frozen=True, slots=True)
class CalibrationRoleManifest:
    role: DataRole
    row_count: int
    row_id_sha256: Sha256


@dataclass(frozen=True, slots=True)
class ClientCalibrationManifest:
    client_id: ClientId
    roles: tuple[CalibrationRoleManifest, ...]

    def role(self, role: DataRole) -> CalibrationRoleManifest:
        for item in self.roles:
            if item.role is role:
                return item
        raise KeyError(role.value)


@dataclass(frozen=True, slots=True)
class CalibrationAssignmentManifest:
    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    clients: tuple[ClientCalibrationManifest, ...]

    def client(self, client_id: ClientId) -> ClientCalibrationManifest:
        for item in self.clients:
            if item.client_id == client_id:
                return item
        raise KeyError(client_id.value)


@dataclass(frozen=True, slots=True)
class RoleArtifactManifest:
    role: DataRole
    rows: int
    row_id_sha256: Sha256
    relative_path: PurePosixPath
    file_sha256: Sha256


@dataclass(frozen=True, slots=True)
class ClientDatasetManifest:
    client_id: ClientId
    roles: tuple[RoleArtifactManifest, ...]

    def role(self, role: DataRole) -> RoleArtifactManifest:
        for item in self.roles:
            if item.role is role:
                return item
        raise KeyError(role.value)


@dataclass(frozen=True, slots=True)
class CalibrationAssignmentReference:
    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    sha256: Sha256


@dataclass(frozen=True, slots=True)
class PreparedDatasetManifest:
    dataset_id: DatasetId
    source_version: str
    parser_version: str
    data_spec_hash: Sha256
    feature_names: tuple[str, ...]
    clients: tuple[ClientDatasetManifest, ...]
    source_files: tuple[SourceFileManifest, ...]
    calibration_assignments: tuple[CalibrationAssignmentReference, ...]
    external_replication_supported: bool
    dataset_level_code: FailureCode | None
    created_at: datetime
    deterministic_payload_sha256: Sha256

    @property
    def client_ids(self) -> tuple[ClientId, ...]:
        return tuple(client.client_id for client in self.clients)

    def client(self, client_id: ClientId) -> ClientDatasetManifest:
        for client in self.clients:
            if client.client_id == client_id:
                return client
        raise KeyError(client_id.value)



def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> Sha256:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return Sha256(digest.hexdigest())



def source_file_manifest(path: Path, root: Path) -> SourceFileManifest:
    return SourceFileManifest(
        relative_path=PurePosixPath(path.relative_to(root).as_posix()),
        sha256=hash_file(path),
        size_bytes=path.stat().st_size,
    )



def hash_row_ids(values: Iterable[RowId | str]) -> Sha256:
    normalized = sorted(
        value.value if isinstance(value, RowId) else str(value)
        for value in values
    )
    payload = "\n".join(normalized).encode("utf-8")
    return Sha256(hashlib.sha256(payload).hexdigest())
