"""Integration tests for running an experiment through the application service."""

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


def test_run_application_creates_immutable_complete_run(tmp_path: Path) -> None:
    """A successful run writes a complete manifest and refuses further mutation."""
    config = _config(tmp_path)
    result, layout = RunExperiment().execute(
        ExperimentId.READINESS_SAMPLE_SIZE,
        config,
        11,
        1000,
        PolicyId.FEDCRG,
        lambda plan, run_layout: None,
        ROOT,
    )
    assert result is None
    assert layout.manifest.exists()
    assert (layout.verification / "hashes.json").exists()
    stored = RunManifestStore().load(layout.manifest)
    assert stored.status is ExperimentStatus.COMPLETE
    summary = layout.reports / "summary.md"
    assert summary.is_file()
    assert "# FedCRG Run" in summary.read_text(encoding="utf-8")
    store = RunManifestStore()
    mutated = stored.model_copy(update={"status": ExperimentStatus.RUNNING})
    with pytest.raises(ImmutableRunError):
        store.save(layout.manifest, mutated)


def test_run_application_records_failed_status(tmp_path: Path) -> None:
    """A run whose work callback raises still records a FAILED manifest before re-raising."""
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
    assert not (runs[0] / "reports" / "summary.md").exists()


def test_run_report_failure_keeps_run_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A summary-report failure never flips a completed run back to FAILED."""
    config = _config(tmp_path)
    service = RunExperiment()

    def boom(run_dir: Path) -> Path:
        raise RuntimeError("report boom")

    monkeypatch.setattr("fedcrg.experiments.runner.build_run_report", boom)
    _, layout = service.execute(
        ExperimentId.READINESS_SAMPLE_SIZE,
        config,
        11,
        1000,
        PolicyId.FEDCRG,
        lambda plan, run_layout: None,
        ROOT,
    )
    assert RunManifestStore().load(layout.manifest).status is ExperimentStatus.COMPLETE
    assert not (layout.root / "reports" / "summary.md").exists()
