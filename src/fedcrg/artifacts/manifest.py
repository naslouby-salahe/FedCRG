"""Typed run manifest and completed-run immutability contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.core.enums import ExperimentId, ExperimentStatus, PolicyId
from fedcrg.core.exceptions import ImmutableRunError
from fedcrg.core.ids import RunId, Sha256


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: RunId
    experiment_id: ExperimentId
    policy_id: PolicyId
    config_hash: Sha256
    model_seed: int
    calibration_seed: int
    status: ExperimentStatus


class RunManifestStore:
    def load(self, path: Path) -> RunManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return RunManifest(
            run_id=RunId(str(raw["run_id"])),
            experiment_id=ExperimentId(raw["experiment_id"]),
            policy_id=PolicyId(raw["policy_id"]),
            config_hash=Sha256(str(raw["config_hash"])),
            model_seed=int(raw["model_seed"]),
            calibration_seed=int(raw["calibration_seed"]),
            status=ExperimentStatus(raw["status"]),
        )

    def save(self, path: Path, manifest: RunManifest) -> None:
        if path.exists() and self.load(path).status is ExperimentStatus.COMPLETE:
            raise ImmutableRunError(f"Completed run is immutable: {path.parent}")
        payload = asdict(manifest)
        payload["run_id"] = manifest.run_id.value
        payload["experiment_id"] = manifest.experiment_id.value
        payload["policy_id"] = manifest.policy_id.value
        payload["status"] = manifest.status.value
        payload["config_hash"] = manifest.config_hash.value
        atomic_write_json(path, payload)
