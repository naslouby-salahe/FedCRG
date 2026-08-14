"""Unit tests for campaign status persistence and runner bookkeeping."""

from __future__ import annotations

from pathlib import Path

import pytest

from fedcrg.config import Study
from fedcrg.experiments.runner import (
    CampaignExecutor,
    CampaignStatus,
    CampaignStatusStore,
    CampaignWorkItem,
    RunAllExperiments,
    WorkloadExecution,
    _CampaignOutcomeRow,
)
from fedcrg.types import CampaignStage, ExperimentId, ExperimentStatus


def _status() -> CampaignStatus:
    return CampaignStatus(
        campaign_id="c1",
        created_at="2026-08-13T00:00:00+0000",
        updated_at="2026-08-13T00:00:01+0000",
        current_experiment=ExperimentId.PRIMARY_NBAIOT,
        current_stage=CampaignStage.RUNNING,
        experiments=(
            _CampaignOutcomeRow(
                experiment_id=ExperimentId.PRIMARY_NBAIOT,
                status=ExperimentStatus.COMPLETE,
                finished_at="2026-08-13T00:00:01+0000",
            ),
        ),
        results_path="results/c1",
        elapsed_seconds=1.5,
    )


def test_status_store_round_trips(tmp_path: Path) -> None:
    store = CampaignStatusStore(campaigns_root=tmp_path)
    status = _status()
    path = store.save(status)
    assert path == tmp_path / "c1.json"
    loaded = store.load("c1")
    assert loaded.campaign_id == "c1"
    assert loaded.completed_experiments == (ExperimentId.PRIMARY_NBAIOT,)
    assert loaded.results_path == "results/c1"
    assert loaded.elapsed_seconds == 1.5


def test_status_store_rejects_unsafe_campaign_ids(tmp_path: Path) -> None:
    store = CampaignStatusStore(campaigns_root=tmp_path)
    with pytest.raises(ValueError):
        store.path_for("../../etc/passwd")


def test_status_store_missing_campaign_raises(tmp_path: Path) -> None:
    store = CampaignStatusStore(campaigns_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("missing")


class _FailingRunner(RunAllExperiments):
    def execute(self, experiment_id, config, prepared_root):  # type: ignore[override]
        if experiment_id is ExperimentId.PRIMARY_NBAIOT:
            raise RuntimeError("boom")
        return WorkloadExecution(experiment_id=experiment_id, models=(), run_directories=())


def test_campaign_runner_records_failures_and_continues(tmp_path: Path) -> None:
    store = CampaignStatusStore(campaigns_root=tmp_path)
    items = (
        CampaignWorkItem(
            experiment_id=ExperimentId.PRIMARY_NBAIOT,
            config_path=Path("config/study.yaml"),
            prepared_root=tmp_path / "preprocessed",
        ),
        CampaignWorkItem(
            experiment_id=ExperimentId.READINESS_SAMPLE_SIZE,
            config_path=Path("config/study.yaml"),
            prepared_root=tmp_path / "preprocessed",
        ),
    )
    status = CampaignExecutor(study=Study.load(), status_store=store, runner=_FailingRunner()).run(
        "c1", items, outputs_root=tmp_path / "outputs"
    )
    assert status.failed_experiments == (ExperimentId.PRIMARY_NBAIOT,)
    assert status.completed_experiments == (ExperimentId.READINESS_SAMPLE_SIZE,)
    assert status.current_stage is CampaignStage.FAILED
