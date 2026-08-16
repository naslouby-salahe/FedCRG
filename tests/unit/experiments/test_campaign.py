"""Tests for campaign status persistence and resumable campaign execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from fedcrg.config import ExperimentConfig, Study
from fedcrg.experiments.execution import ExperimentRunResult
from fedcrg.experiments.runner import (
    CampaignExecutor,
    CampaignStatus,
    CampaignStatusStore,
    CampaignWorkItem,
    RunAllExperiments,
    WorkloadExecution,
    CampaignOutcomeRow,
)
from fedcrg.types import (
    CalibrationSeed,
    CampaignStage,
    CompletionState,
    ExecutionOutcome,
    ExperimentId,
    ExperimentStatus,
)


def _status() -> CampaignStatus:
    return CampaignStatus(
        created_at="2026-08-13T00:00:00+0000",
        updated_at="2026-08-13T00:00:01+0000",
        current_experiment=ExperimentId.PRIMARY_NBAIOT,
        current_stage=CampaignStage.RUNNING,
        experiments=(
            CampaignOutcomeRow(
                experiment_id=ExperimentId.PRIMARY_NBAIOT,
                status=ExperimentStatus.COMPLETE,
                finished_at="2026-08-13T00:00:01+0000",
            ),
        ),
        results_path="results",
        elapsed_seconds=1.5,
    )


def test_status_store_round_trips(tmp_path: Path) -> None:
    store = CampaignStatusStore(campaigns_root=tmp_path)
    status = _status()
    path = store.save(status)
    assert path == tmp_path / "status.json"
    loaded = store.load()
    assert loaded.completed_experiments == (ExperimentId.PRIMARY_NBAIOT,)
    assert loaded.results_path == "results"
    assert loaded.elapsed_seconds == 1.5


def test_status_store_missing_status_raises(tmp_path: Path) -> None:
    store = CampaignStatusStore(campaigns_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load()


class _FailingRunner(RunAllExperiments):
    def execute(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        prepared_root: Path,
        *,
        calibration_seeds: tuple[CalibrationSeed, ...] | None = None,
    ) -> WorkloadExecution:
        if experiment_id is ExperimentId.PRIMARY_NBAIOT:
            raise RuntimeError("boom")
        return WorkloadExecution(experiment_id=experiment_id, models=(), run_directories=())


class _ScriptedExecutor:
    def execute(
        self, experiment_id: ExperimentId, *, overwrite: bool = False
    ) -> ExperimentRunResult:
        if experiment_id is ExperimentId.PRIMARY_NBAIOT:
            raise RuntimeError("boom")
        return ExperimentRunResult(
            experiment_id=experiment_id,
            outcome=ExecutionOutcome.COMPLETED,
            state=CompletionState.FULLY_PASSED,
            json_paths=(),
            csv_paths=(),
            figure_paths=(),
            report_paths=(),
            bundle_path="results/experiments/readiness_sample_size",
            output_root="outputs",
            model_count=0,
            run_directory_count=0,
        )


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
    status = CampaignExecutor(
        study=Study.load(),
        status_store=store,
        runner=_FailingRunner(),
        executor=_ScriptedExecutor(),
    ).run(items, outputs_root=tmp_path / "outputs")
    assert status.failed_experiments == (ExperimentId.PRIMARY_NBAIOT,)
    assert status.completed_experiments == ()
    assert status.experiments[-1].status is ExperimentStatus.BLOCKED
    assert status.current_stage is CampaignStage.FAILED
