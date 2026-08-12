from pathlib import Path

import pytest

from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.manifest import RunManifest, RunManifestStore
from fedcrg.core.enums import ExperimentStatus
from fedcrg.core.exceptions import ImmutableRunError
from fedcrg.core.ids import RunId


def test_completed_run_manifest_is_immutable(tmp_path: Path) -> None:
    layout = RunLayout.for_run(tmp_path, RunId("run-1"))
    layout.create()
    store = RunManifestStore()
    complete = RunManifest("run-1", "primary_nbaiot", "h", 11, 1000, ExperimentStatus.COMPLETE)
    store.save(layout.manifest, complete)
    with pytest.raises(ImmutableRunError):
        store.save(layout.manifest, RunManifest("run-1", "primary_nbaiot", "h", 11, 1000, ExperimentStatus.RUNNING))
