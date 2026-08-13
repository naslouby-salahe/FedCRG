"""Verified references from immutable run evidence to reusable cache artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from fedcrg.artifacts.hashing import sha256_file
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.domain.identifiers import Sha256


@dataclass(frozen=True, slots=True)
class CacheReference:
    """Reference one immutable cache file without duplicating it per policy cell."""

    relative_path: PurePosixPath
    sha256: Sha256

    @classmethod
    def build(cls, file_path: Path, outputs_root: Path) -> CacheReference:
        resolved_file = file_path.resolve()
        resolved_outputs = outputs_root.resolve()
        try:
            relative = resolved_file.relative_to(resolved_outputs)
        except ValueError as exc:
            raise ValueError(
                f"Cache artifact must live under outputs root: {resolved_file}"
            ) from exc
        return cls(PurePosixPath(relative.as_posix()), Sha256(sha256_file(resolved_file)))

    def resolve(self, outputs_root: Path) -> Path:
        path = (outputs_root / Path(self.relative_path)).resolve()
        try:
            path.relative_to(outputs_root.resolve())
        except ValueError as exc:
            raise ValueError("Cache reference escapes the outputs root") from exc
        return path

    def verify(self, outputs_root: Path) -> bool:
        path = self.resolve(outputs_root)
        return path.is_file() and Sha256(sha256_file(path)) == self.sha256


class CacheReferenceStore:
    def save(self, path: Path, reference: CacheReference) -> None:
        atomic_write_json(path, reference)

    def load(self, path: Path) -> CacheReference:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CacheReference(
            relative_path=PurePosixPath(str(payload["relative_path"])),
            sha256=Sha256(str(payload["sha256"])),
        )
