"""Dataset-adapter contract, discovery, and prepared-cache provenance manifests."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

import pandas as pd

from fedcrg.domain.enums import (
    CalibrationAssignmentMode,
    ChronologyStatus,
    DataRole,
    DatasetId,
    FailureCode,
)
from fedcrg.domain.errors import DataIntegrityError
from fedcrg.domain.identifiers import CalibrationSeed, ClientId, RowId, Sha256


class DatasetDiscovery:
    """Resolve dataset files without embedding workstation-specific paths."""

    @staticmethod
    def require_root(root: Path) -> Path:
        resolved = root.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        return resolved

    @staticmethod
    def directories(root: Path) -> tuple[Path, ...]:
        resolved = DatasetDiscovery.require_root(root)
        return tuple(sorted(path for path in resolved.iterdir() if path.is_dir()))

    @staticmethod
    def csv_files(root: Path, recursive: bool = True) -> tuple[Path, ...]:
        resolved = DatasetDiscovery.require_root(root)
        pattern = "**/*.csv" if recursive else "*.csv"
        files = tuple(sorted(path for path in resolved.glob(pattern) if path.is_file()))
        if not files:
            raise DataIntegrityError(f"No CSV files found under {resolved}")
        return files


@dataclass(frozen=True, slots=True)
class ClientData:
    dataset: DatasetId
    client_id: ClientId
    benign: pd.DataFrame
    attack: pd.DataFrame
    chronology: ChronologyStatus = ChronologyStatus.SOURCE_ORDER_ONLY


class DatasetAdapter(ABC):
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()

    @property
    @abstractmethod
    def dataset_id(self) -> DatasetId:
        raise NotImplementedError

    @abstractmethod
    def discover_clients(self) -> tuple[ClientId, ...]:
        raise NotImplementedError

    @abstractmethod
    def load_client(self, client_id: ClientId) -> ClientData:
        raise NotImplementedError

    @abstractmethod
    def source_files(self) -> tuple[Path, ...]:
        raise NotImplementedError


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> Sha256:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return Sha256(digest.hexdigest())


def hash_row_ids(values: Iterable[RowId | str]) -> Sha256:
    normalized = sorted(value.value if isinstance(value, RowId) else str(value) for value in values)
    payload = "\n".join(normalized).encode("utf-8")
    return Sha256(hashlib.sha256(payload).hexdigest())


@dataclass(frozen=True, slots=True)
class SourceFileManifest:
    relative_path: PurePosixPath
    sha256: Sha256
    size_bytes: int


def source_file_manifest(path: Path, root: Path) -> SourceFileManifest:
    return SourceFileManifest(
        relative_path=PurePosixPath(path.relative_to(root).as_posix()),
        sha256=hash_file(path),
        size_bytes=path.stat().st_size,
    )


@dataclass(frozen=True, slots=True)
class CalibrationAssignmentReference:
    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    sha256: Sha256


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
        raise KeyError(role)


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
