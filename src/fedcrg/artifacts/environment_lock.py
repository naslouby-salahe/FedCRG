"""Freeze the validated Python environment without depending on a package manager CLI."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from fedcrg.artifacts.hashing import sha256_file
from fedcrg.artifacts.serialization import atomic_write_text
from fedcrg.domain.identifiers import Sha256


_DEFAULT_DISTRIBUTIONS = (
    "click",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "pydantic",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "torch",
)


@dataclass(frozen=True, slots=True)
class EnvironmentLock:
    path: Path
    sha256: Sha256
    distributions: tuple[str, ...]


class EnvironmentLocker:
    """Write exact installed versions after the first validated protocol environment."""

    def freeze(
        self,
        path: Path,
        distributions: tuple[str, ...] = _DEFAULT_DISTRIBUTIONS,
    ) -> EnvironmentLock:
        if path.exists():
            raise FileExistsError(
                f"Environment lock already exists and must be versioned rather than overwritten: {path}"
            )
        normalized: list[tuple[str, str]] = []
        for distribution in distributions:
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"Required distribution is not installed: {distribution}"
                ) from exc
            canonical_name = metadata.metadata(distribution).get("Name", distribution)
            normalized.append((canonical_name, version))
        normalized.sort(key=lambda item: item[0].lower())
        content = "".join(f"{name}=={version}\n" for name, version in normalized)
        atomic_write_text(path, content)
        return EnvironmentLock(
            path=path,
            sha256=Sha256(sha256_file(path)),
            distributions=tuple(name for name, _ in normalized),
        )
