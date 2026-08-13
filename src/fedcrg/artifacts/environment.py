"""Reproducibility environment and repository-state capture."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import numpy
import pandas
import scipy
import sklearn
import torch

from fedcrg.artifacts.hashing import sha256_file


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_environment(repository_root: Path = Path(".")) -> dict[str, object]:
    commit = _git(["rev-parse", "HEAD"], repository_root)
    dirty = _git(["status", "--porcelain"], repository_root)
    patch = _git(["diff", "--binary"], repository_root) if dirty else ""
    lock_candidates = (repository_root / "uv.lock", repository_root / "requirements.lock")
    lock_path = next((path for path in lock_candidates if path.exists()), None)
    return {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_device_count": torch.cuda.device_count(),
        "git_commit": commit,
        "git_clean": dirty == "" if dirty is not None else None,
        "git_patch_sha256": __import__("hashlib").sha256((patch or "").encode()).hexdigest() if dirty else None,
        "environment_lock": str(lock_path) if lock_path else None,
        "environment_lock_sha256": sha256_file(lock_path) if lock_path else None,
    }
