"""Typed run manifest and completed-run immutability contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.core.enums import ExperimentId, ExperimentStatus, PolicyId
from fedcrg.core.exceptions import ImmutableRunError
from fedcrg.core.ids import CalibrationSeed, ModelSeed, RunId, Sha256


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: RunId
    experiment_id: ExperimentId
    policy_id: PolicyId
    config_hash: Sha256
    model_seed: ModelSeed
    calibration_seed: CalibrationSeed
    status: ExperimentStatus


class RunManifestStore:
    def load(self, path: Path) -> RunManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return RunManifest(
            run_id=RunId(str(raw["run_id"])),
            experiment_id=ExperimentId(str(raw["experiment_id"])),
            policy_id=PolicyId(str(raw["policy_id"])),
            config_hash=Sha256(str(raw["config_hash"])),
            model_seed=ModelSeed(int(raw["model_seed"])),
            calibration_seed=CalibrationSeed(int(raw["calibration_seed"])),
            status=ExperimentStatus(str(raw["status"])),
        )

    def save(self, path: Path, manifest: RunManifest) -> None:
        if path.exists() and self.load(path).status is ExperimentStatus.COMPLETE:
            raise ImmutableRunError(f"Completed run is immutable: {path.parent}")
        atomic_write_json(path, manifest)
