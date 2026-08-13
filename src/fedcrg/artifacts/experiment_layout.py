"""Structured evidence layout for experiment-level outputs that are not policy runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentArtifactLayout:
    root: Path
    protocol_code: str

    @classmethod
    def for_code(cls, outputs_root: Path, protocol_code: str) -> "ExperimentArtifactLayout":
        return cls(outputs_root / "experiments" / protocol_code, protocol_code)

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def results(self) -> Path:
        return self.root / "results.json"

    @property
    def cells(self) -> Path:
        return self.root / "cells"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.cells.mkdir(exist_ok=True)
        self.figures.mkdir(exist_ok=True)
        self.tables.mkdir(exist_ok=True)
