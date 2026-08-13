"""Reproducibility environment capture and validated environment locking."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy
import pandas
import scipy
import sklearn
import torch

from fedcrg.artifacts.integrity import sha256_file
from fedcrg.artifacts.json_io import atomic_write_text
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


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Typed reproducibility snapshot, serialized only at the JSON boundary."""

    python: str
    pytorch: str
    cuda_runtime: str | None
    cudnn: str | int | None
    numpy: str
    scipy: str
    pandas: str
    scikit_learn: str
    os: str
    cpu: str
    gpu: str | None
    cuda_device_count: int
    git_commit: str | None
    git_clean: bool | None
    git_patch_sha256: str | None
    environment_pin_kind: str | None
    environment_pin_sha256: str | None


def capture_environment(repository_root: Path = Path(".")) -> EnvironmentSnapshot:
    commit = _git(["rev-parse", "HEAD"], repository_root)
    dirty = _git(["status", "--porcelain"], repository_root)
    patch = _git(["diff", "--binary"], repository_root) if dirty else ""
    lock_candidates = (repository_root / "uv.lock", repository_root / "requirements.lock")
    lock_path = next((path for path in lock_candidates if path.exists()), None)
    return EnvironmentSnapshot(
        python=sys.version,
        pytorch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        pandas=pandas.__version__,
        scikit_learn=sklearn.__version__,
        os=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cuda_device_count=torch.cuda.device_count(),
        git_commit=commit,
        git_clean=dirty == "" if dirty is not None else None,
        git_patch_sha256=hashlib.sha256((patch or "").encode()).hexdigest() if dirty else None,
        environment_pin_kind=lock_path.name if lock_path else None,
        environment_pin_sha256=sha256_file(lock_path) if lock_path else None,
    )


@dataclass(frozen=True, slots=True)
class PackagePin:
    """One pinned distribution as ``name==version`` for the environment lock."""

    name: str
    version: str

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("Package pin requires both name and version")

    @property
    def requirement(self) -> str:
        return f"{self.name}=={self.version}"


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
        pins: list[PackagePin] = []
        for distribution in distributions:
            try:
                version = metadata.version(distribution)
            except metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"Required distribution is not installed: {distribution}"
                ) from exc
            package_metadata = metadata.metadata(distribution)
            raw_name = package_metadata["Name"] if "Name" in package_metadata else distribution
            pins.append(PackagePin(name=str(raw_name), version=version))
        pins.sort(key=lambda pin: pin.name.lower())
        content = "".join(f"{pin.requirement}\n" for pin in pins)
        atomic_write_text(path, content)
        return EnvironmentLock(
            path=path,
            sha256=Sha256(sha256_file(path)),
            distributions=tuple(pin.name for pin in pins),
        )
