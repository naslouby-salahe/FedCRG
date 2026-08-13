"""Unit tests for campaign status persistence and runner bookkeeping."""

from __future__ import annotations

from pathlib import Path

import pytest

from fedcrg.domain.enums import CampaignStatusValue, ExperimentId
from fedcrg.domain.errors import ConfigurationError
from fedcrg.experiments.campaign import (
    CampaignRunner,
    CampaignStatus,
    CampaignStatusStore,
    CampaignWorkItem,
    ExperimentCampaignStatus,
)


def test_status_store_round_trips(tmp_path: Path) -> None:
    store = CampaignStatusStore(tmp_path)
    status = CampaignStatus(
        campaign_id="c1",
        created_at="2026-08-13T00:00:00+0000",
        updated_at="2026-08-13T00:00:01+0000",
        current_experiment="primary_nbaiot",
        current_stage="complete primary_nbaiot",
        experiments=(
            ExperimentCampaignStatus(
                experiment_id=ExperimentId.PRIMARY_NBAIOT,
                status=CampaignStatusValue.COMPLETE,
                started_at="2026-08-13T00:00:00+0000",
                finished_at="2026-08-13T00:00:01+0000",
                run_directories=("outputs/runs/x",),
            ),
            ExperimentCampaignStatus(
                experiment_id=ExperimentId.EXTERNAL_DIAD,
                status=CampaignStatusValue.BLOCKED,
                problem="dependency failed",
            ),
        ),
        results_path="results/c1",
        elapsed_seconds=1.5,
    )
    path = store.save(status)
    assert path == tmp_path / "c1.json"

    loaded = store.load("c1")
    assert loaded.campaign_id == "c1"
    assert loaded.completed_experiments == ("primary_nbaiot",)
    assert loaded.blocked_experiments == ("external_diad",)
    assert loaded.experiments[0].run_directories == ("outputs/runs/x",)
    assert loaded.results_path == "results/c1"
    assert loaded.elapsed_seconds == 1.5


def test_status_store_rejects_unsafe_campaign_ids(tmp_path: Path) -> None:
    store = CampaignStatusStore(tmp_path)
    with pytest.raises(ConfigurationError):
        store.path_for("../../etc/passwd")


def test_status_store_missing_campaign_raises(tmp_path: Path) -> None:
    store = CampaignStatusStore(tmp_path)
    with pytest.raises(ConfigurationError):
        store.load("missing")


def test_campaign_runner_marks_blocked_when_dependency_fails(tmp_path: Path) -> None:
    store = CampaignStatusStore(tmp_path)

    class FailingRunner(CampaignRunner):
        def _execute_item(self, item: CampaignWorkItem) -> tuple[Path, ...]:
            if item.experiment_id is ExperimentId.PRIMARY_NBAIOT:
                raise RuntimeError("boom")
            return ()

        def _record_telemetry(self, outputs_root: Path) -> None:
            return None

    items = (
        CampaignWorkItem(ExperimentId.PRIMARY_NBAIOT, tmp_path / "a.yaml"),
        CampaignWorkItem(ExperimentId.READINESS_SAMPLE_SIZE, tmp_path / "b.yaml"),
    )
    runner = FailingRunner(store=store)
    with pytest.raises(RuntimeError, match="failed"):
        runner.run("c1", items, outputs_root=tmp_path / "outputs")
    status = store.load("c1")
    assert status.failed_experiments == ("primary_nbaiot",)
    assert status.blocked_experiments == ("readiness_sample_size",)
