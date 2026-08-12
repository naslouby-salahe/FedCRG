"""Dataset and split provenance manifests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from fedcrg.core.enums import DataRole, DatasetId


@dataclass(frozen=True, slots=True)
class SourceFileManifest:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset: DatasetId
    feature_count: int
    clients: tuple[str, ...]
    files: tuple[SourceFileManifest, ...]


@dataclass(frozen=True, slots=True)
class SplitManifest:
    dataset: DatasetId
    client_id: str
    calibration_seed: int
    row_counts: dict[DataRole, int]
    row_id_hashes: dict[DataRole, str]


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_row_ids(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
