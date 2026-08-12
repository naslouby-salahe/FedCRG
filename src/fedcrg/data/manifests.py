"""Dataset, eligibility, and split provenance manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from fedcrg.core.enums import ChronologyStatus, DataRole, DatasetId
from fedcrg.core.ids import ClientId, Sha256
from fedcrg.data.models import EligibilityRecord


@dataclass(frozen=True, slots=True)
class SourceFileManifest:
    relative_path: str
    sha256: Sha256
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: DatasetId
    source_version: str
    parser_version: str
    feature_names: tuple[str, ...]
    clients: tuple[ClientId, ...]
    files: tuple[SourceFileManifest, ...]
    per_role_counts: dict[ClientId, dict[DataRole, int]]


@dataclass(frozen=True, slots=True)
class EligibilityManifest:
    dataset_id: DatasetId
    discovered_clients: tuple[ClientId, ...]
    eligible_clients: tuple[ClientId, ...]
    records: tuple[EligibilityRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id.value,
            "discovered_clients": [item.value for item in self.discovered_clients],
            "eligible_clients": [item.value for item in self.eligible_clients],
            "records": [
                {
                    "client_id": item.client_id.value,
                    "status": item.status.value,
                    "benign_count": item.benign_count,
                    "malicious_count": item.malicious_count,
                    "attack_development_capacity": item.attack_development_capacity,
                    "primary_code": None if item.primary_code is None else item.primary_code.value,
                    "secondary_codes": [code.value for code in item.secondary_codes],
                    "chronology": item.chronology.value,
                }
                for item in self.records
            ],
        }


@dataclass(frozen=True, slots=True)
class SplitManifest:
    dataset: DatasetId
    client_id: ClientId
    calibration_seed: int
    row_counts: dict[DataRole, int]
    row_id_hashes: dict[DataRole, Sha256]
    chronology: ChronologyStatus


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_manifest(path: Path, root: Path) -> SourceFileManifest:
    return SourceFileManifest(
        relative_path=path.relative_to(root).as_posix(),
        sha256=Sha256(hash_file(path)),
        size_bytes=path.stat().st_size,
    )


def hash_row_ids(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CalibrationRoleManifest:
    row_count: int
    row_id_sha256: Sha256

    def to_dict(self) -> dict[str, object]:
        return {"row_count": self.row_count, "row_id_sha256": self.row_id_sha256.value}


@dataclass(frozen=True, slots=True)
class ClientCalibrationManifest:
    client_id: ClientId
    roles: dict[DataRole, CalibrationRoleManifest]

    def to_dict(self) -> dict[str, object]:
        return {
            "client_id": self.client_id.value,
            "roles": {role.value: item.to_dict() for role, item in self.roles.items()},
        }


@dataclass(frozen=True, slots=True)
class CalibrationAssignmentManifest:
    calibration_seed: int
    mode: str
    clients: tuple[ClientCalibrationManifest, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_seed": self.calibration_seed,
            "mode": self.mode,
            "clients": [client.to_dict() for client in self.clients],
        }
