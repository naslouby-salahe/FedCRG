"""Integration tests for the immutable run-output layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from fedcrg.evidence.models import RunManifest
from fedcrg.evidence.store import RunLayout, RunManifestStore
from fedcrg.types import ExperimentId, ExperimentStatus, ImmutableRunError, PolicyId


def test_completed_run_manifest_is_immutable(tmp_path: Path) -> None:
    layout = RunLayout.for_run(tmp_path, "run-1")
    layout.create()
    store = RunManifestStore()

    def _manifest(status: ExperimentStatus) -> RunManifest:
        return RunManifest(
            run_id="run-1",
            experiment_id=ExperimentId.PRIMARY_NBAIOT,
            policy_id=PolicyId.FEDCRG,
            config_hash="a" * 64,
            model_seed=11,
            calibration_seed=1000,
            status=status,
        )

    complete = _manifest(ExperimentStatus.COMPLETE)
    store.save(layout.manifest, complete)
    with pytest.raises(ImmutableRunError):
        store.save(layout.manifest, _manifest(ExperimentStatus.RUNNING))
