from __future__ import annotations

from pathlib import Path

import pytest

from fedcrg.config import ExperimentConfig
from fedcrg.evidence.store import RunManifestStore
from fedcrg.experiments.runner import ExperimentPlan, RunExperiment
from fedcrg.paths import RunLayout
from fedcrg.types import (
    ExperimentId,
    ExperimentStatus,
    ImmutableRunError,
    PolicyId,
)
from tests._fixtures import primary_experiment_config

ROOT = Path(__file__).resolve().parents[2]


def _config(root: Path) -> ExperimentConfig:
    return primary_experiment_config(root, ExperimentId.READINESS_SAMPLE_SIZE)


def _write_required_run_files(layout: RunLayout) -> None:
    for path in (
        layout.dataset_manifest,
        layout.preprocessing_evidence,
        layout.training_manifest,
        layout.model_reference,
        layout.score_manifest,
        layout.threshold_records,
        layout.metric_records,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")


def test_run_application_creates_immutable_complete_run(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def write_files(plan: ExperimentPlan, run_layout: RunLayout) -> None:
        _write_required_run_files(run_layout)

    result, layout = RunExperiment().execute(
        ExperimentId.READINESS_SAMPLE_SIZE,
        config,
        11,
        1000,
        PolicyId.FEDCRG,
        write_files,
        ROOT,
    )
    assert result is None
    assert layout.manifest.exists()
    assert (layout.verification / "hashes.json").exists()
    stored = RunManifestStore().load(layout.manifest)
    assert stored.status is ExperimentStatus.COMPLETE
    store = RunManifestStore()
    mutated = stored.model_copy(update={"status": ExperimentStatus.RUNNING})
    with pytest.raises(ImmutableRunError):
        store.save(layout.manifest, mutated)


def test_run_application_records_failed_status(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = RunExperiment()

    def fail(plan: ExperimentPlan, run_layout: RunLayout) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        service.execute(
            ExperimentId.READINESS_SAMPLE_SIZE,
            config,
            11,
            1000,
            PolicyId.FEDCRG,
            fail,
            ROOT,
        )
    runs = list((tmp_path / "runs").iterdir())
    assert RunManifestStore().load(runs[0] / "manifest.json").status is ExperimentStatus.FAILED
