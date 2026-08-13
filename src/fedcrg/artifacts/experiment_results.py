"""Typed envelopes for experiment-level evidence outside policy run directories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.core.ids import Sha256


@dataclass(frozen=True, slots=True)
class ExperimentResultEnvelope:
    """Self-describing aggregate evidence for one pre-registered S/R experiment."""

    protocol_code: str
    config_hash: Sha256
    master_seed: int | None
    expected_cells: int | None
    expected_monte_carlo_trials: int
    expected_exact_cells: int
    cells: tuple[dict[str, object], ...]
    notes: tuple[str, ...] = ()
    metadata: dict[str, object] | None = None

    @property
    def observed_cells(self) -> int:
        return len(self.cells)

    @property
    def observed_monte_carlo_trials(self) -> int:
        total = 0
        for cell in self.cells:
            repetitions = cell.get("repetitions")
            if isinstance(repetitions, int):
                total += repetitions
        return total

    @property
    def complete(self) -> bool:
        if self.expected_cells is not None and self.observed_cells != self.expected_cells:
            return False
        if (
            self.expected_monte_carlo_trials
            and self.observed_monte_carlo_trials != self.expected_monte_carlo_trials
        ):
            return False
        if self.expected_exact_cells and self.observed_cells != self.expected_exact_cells:
            return False
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "experiment": self.protocol_code,
            "config_hash": self.config_hash.value,
            "master_seed": self.master_seed,
            "expected_cells": self.expected_cells,
            "observed_cells": self.observed_cells,
            "expected_monte_carlo_trials": self.expected_monte_carlo_trials,
            "observed_monte_carlo_trials": self.observed_monte_carlo_trials,
            "expected_exact_cells": self.expected_exact_cells,
            "complete": self.complete,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": list(self.notes),
            "metadata": self.metadata or {},
            "cells": list(self.cells),
        }

    def write(self, path: Path) -> Path:
        atomic_write_json(path, self.to_dict())
        return path


@dataclass(frozen=True, slots=True)
class ExperimentCellEnvelope:
    """Evidence for one model/calibration cell of a real-score sensitivity."""

    protocol_code: str
    config_hash: Sha256
    model_seed: int
    calibration_seed: int
    expected_subcells: int | None
    dataset_id: str
    cells: tuple[dict[str, object], ...]
    score_cache_sha256: Sha256
    data_spec_hash: Sha256
    training_spec_hash: Sha256
    notes: tuple[str, ...] = ()
    metadata: dict[str, object] | None = None

    @property
    def complete(self) -> bool:
        return self.expected_subcells is None or len(self.cells) == self.expected_subcells

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "experiment": self.protocol_code,
            "config_hash": self.config_hash.value,
            "dataset_id": self.dataset_id,
            "model_seed": self.model_seed,
            "calibration_seed": self.calibration_seed,
            "score_cache_sha256": self.score_cache_sha256.value,
            "data_spec_hash": self.data_spec_hash.value,
            "training_spec_hash": self.training_spec_hash.value,
            "expected_subcells": self.expected_subcells,
            "observed_subcells": len(self.cells),
            "complete": self.complete,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "notes": list(self.notes),
            "metadata": self.metadata or {},
            "cells": list(self.cells),
        }

    def write(self, path: Path) -> Path:
        atomic_write_json(path, self.to_dict())
        return path
