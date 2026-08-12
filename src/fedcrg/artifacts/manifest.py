"""Run manifest and immutability contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.core.enums import ExperimentStatus
from fedcrg.core.exceptions import ImmutableRunError


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    experiment_id: str
    config_hash: str
    model_seed: int
    calibration_seed: int
    status: ExperimentStatus


class RunManifestStore:
    def load(self, path: Path) -> RunManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["status"] = ExperimentStatus(raw["status"])
        return RunManifest(**raw)

    def save(self, path: Path, manifest: RunManifest) -> None:
        if path.exists():
            current = self.load(path)
            if current.status is ExperimentStatus.COMPLETE:
                raise ImmutableRunError(f"Completed run is immutable: {path.parent}")
        payload = asdict(manifest)
        payload["status"] = manifest.status.value
        atomic_write_json(path, payload)
