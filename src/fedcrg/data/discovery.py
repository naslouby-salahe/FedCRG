"""Filesystem discovery helpers for dataset adapters."""

from __future__ import annotations

from pathlib import Path

from fedcrg.core.exceptions import DataIntegrityError


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
