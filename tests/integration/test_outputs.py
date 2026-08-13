from pathlib import Path

import pytest

from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.manifest import RunManifest, RunManifestStore
from fedcrg.domain.enums import ExperimentId, ExperimentStatus, PolicyId
from fedcrg.domain.errors import ImmutableRunError
from fedcrg.domain.identifiers import CalibrationSeed, ModelSeed, RunId, Sha256


def test_completed_run_manifest_is_immutable(tmp_path: Path) -> None:
    layout = RunLayout.for_run(tmp_path, RunId("run-1"))
    layout.create()
    store = RunManifestStore()

    def _manifest(status: ExperimentStatus) -> RunManifest:
        return RunManifest(
            run_id=RunId("run-1"),
            experiment_id=ExperimentId.PRIMARY_NBAIOT,
            policy_id=PolicyId.FEDCRG,
            config_hash=Sha256("a" * 64),
            model_seed=ModelSeed(11),
            calibration_seed=CalibrationSeed(1000),
            status=status,
        )

    complete = _manifest(ExperimentStatus.COMPLETE)
    store.save(layout.manifest, complete)
    with pytest.raises(ImmutableRunError):
        store.save(layout.manifest, _manifest(ExperimentStatus.RUNNING))
