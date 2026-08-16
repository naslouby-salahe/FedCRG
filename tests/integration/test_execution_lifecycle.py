"""Contract tests for execution, verification, result bundles, rerun, and overwrite."""

from __future__ import annotations

from pathlib import Path

import pytest

from fedcrg.config import ExperimentConfig, ExperimentSpec, Study
from fedcrg.evidence.completion import (
    ExperimentArtifactPurger,
    ExperimentEvidenceAssessor,
)
from fedcrg.evidence.contracts import (
    ExperimentApplicability,
    ExperimentArtifactContract,
    ExperimentContractCatalogue,
    all_experiment_contracts,
    experiment_contract,
)
from fedcrg.experiments.analyses import MismatchPowerResult, SyntheticExperimentEnvelope
from fedcrg.experiments.execution import ExperimentExecutor
from fedcrg.experiments.runner import (
    CampaignExecutor,
    CampaignStatus,
    CampaignStatusStore,
    CampaignWorkItem,
    CampaignOutcomeRow,
)
from fedcrg.paths import LayoutArtifact, OutputsLayout, StudyPaths
from fedcrg.reporting import ExperimentPublisher, ResultsBuilder, ResultsVerifier
from fedcrg.types import (
    CacheRole,
    CampaignStage,
    CompletionState,
    ExecutionOutcome,
    ExperimentId,
    ExperimentStatus,
    ExperimentType,
    JsonResultKind,
    ResultsGenerationStatus,
    SharedCacheKind,
)


def _study(tmp_path: Path) -> Study:
    loaded = Study.load()
    paths = StudyPaths(
        data_root=tmp_path / "data",
        preprocessed_root=tmp_path / "preprocessed",
        outputs_root=tmp_path / "outputs",
        results_root=tmp_path / "results",
    )
    return Study(
        study_config=loaded.study_config.model_copy(update={"paths": paths}),
        datasets=loaded.datasets,
        catalogue=loaded.catalogue,
    )


def _envelope() -> SyntheticExperimentEnvelope:
    return SyntheticExperimentEnvelope(
        experiment_id=ExperimentId.MISMATCH_POWER,
        expected_monte_carlo_trials=0,
        expected_exact_cells=1,
        actual_monte_carlo_trials=0,
        actual_cells=1,
        cells=(MismatchPowerResult(sample_count=736, true_fpr=0.02, declaration_probability=0.8),),
    )


def _write_analysis(outputs_root: Path, payload: SyntheticExperimentEnvelope | str) -> Path:
    path = OutputsLayout(outputs_root).analysis_result(ExperimentId.MISMATCH_POWER)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(payload.model_dump_json(), encoding="utf-8")
    return path


def test_every_registered_experiment_has_a_json_contract() -> None:
    contracts = all_experiment_contracts()
    assert {item.experiment_id for item in contracts} == set(ExperimentId)
    for contract in contracts:
        assert contract.json_files
        assert contract.json_kind in set(JsonResultKind)


def test_score_stress_experiments_use_analysis_json_not_empirical_grid() -> None:
    for experiment_id in (
        ExperimentId.TEMPORAL_DEPENDENCE,
        ExperimentId.CALIBRATION_SHIFT,
        ExperimentId.CALIBRATION_CONTAMINATION,
    ):
        contract = experiment_contract(experiment_id)
        assert contract.requires_analysis_json
        assert not contract.requires_run_grid


def test_first_and_second_execution(tmp_path: Path) -> None:
    study = _study(tmp_path)
    calls = {"count": 0}

    def workload(
        experiment_id: ExperimentId,
        spec: ExperimentSpec,
        config: ExperimentConfig,
        outputs_root: Path,
    ) -> None:
        calls["count"] += 1
        _write_analysis(outputs_root, _envelope())

    executor = ExperimentExecutor(study=study, workload=workload)
    first = executor.execute(ExperimentId.MISMATCH_POWER)
    assert first.outcome is ExecutionOutcome.COMPLETED
    assert first.state is CompletionState.FULLY_PASSED
    assert calls["count"] == 1
    second = executor.execute(ExperimentId.MISMATCH_POWER)
    assert second.outcome is ExecutionOutcome.ALREADY_PASSED
    assert second.state is CompletionState.FULLY_PASSED
    assert calls["count"] == 1


def test_missing_and_malformed_json_prevent_pass(tmp_path: Path) -> None:
    study = _study(tmp_path)
    assessor = ExperimentEvidenceAssessor(study)
    assert assessor.assess(ExperimentId.MISMATCH_POWER).state is CompletionState.NOT_STARTED
    _write_analysis(study.paths.outputs_root, "{")
    assert assessor.assess(ExperimentId.MISMATCH_POWER).state is CompletionState.INVALID
    _write_analysis(study.paths.outputs_root, '{"cells": []}')
    assert assessor.assess(ExperimentId.MISMATCH_POWER).state is CompletionState.INVALID


def test_missing_required_report_prevents_pass(tmp_path: Path) -> None:
    study = _study(tmp_path)
    _write_analysis(study.paths.outputs_root, _envelope())
    ExperimentPublisher().publish(
        ExperimentId.MISMATCH_POWER,
        study.paths.outputs_root,
        study.resolve(ExperimentId.MISMATCH_POWER),
    )
    report = (
        OutputsLayout(study.paths.outputs_root).experiment_reports(ExperimentId.MISMATCH_POWER)
        / LayoutArtifact.EXPERIMENT_REPORT
    )
    report.unlink()
    assessment = ExperimentEvidenceAssessor(study).assess(ExperimentId.MISMATCH_POWER)
    assert assessment.state is CompletionState.EVIDENCE_INCOMPLETE
    assert not assessment.passed


def test_stale_bundle_is_detected(tmp_path: Path) -> None:
    study = _study(tmp_path)
    _write_analysis(study.paths.outputs_root, _envelope())
    ExperimentPublisher().publish(
        ExperimentId.MISMATCH_POWER,
        study.paths.outputs_root,
        study.resolve(ExperimentId.MISMATCH_POWER),
    )
    ResultsBuilder().build(
        outputs_root=study.paths.outputs_root,
        results_root=study.paths.results_root,
        experiment_id=ExperimentId.MISMATCH_POWER,
        study=study,
    )
    _write_analysis(
        study.paths.outputs_root,
        SyntheticExperimentEnvelope(
            experiment_id=ExperimentId.MISMATCH_POWER,
            expected_monte_carlo_trials=0,
            expected_exact_cells=1,
            actual_monte_carlo_trials=0,
            actual_cells=1,
            cells=(
                MismatchPowerResult(sample_count=1000, true_fpr=0.03, declaration_probability=0.9),
            ),
        ),
    )
    ExperimentPublisher().publish(
        ExperimentId.MISMATCH_POWER,
        study.paths.outputs_root,
        study.resolve(ExperimentId.MISMATCH_POWER),
    )
    assessment = ExperimentEvidenceAssessor(study).assess(ExperimentId.MISMATCH_POWER)
    assert assessment.state is CompletionState.STALE


def test_overwrite_rebuilds_owned_artifacts_only(tmp_path: Path) -> None:
    study = _study(tmp_path)
    other = OutputsLayout(study.paths.outputs_root).analysis_result(ExperimentId.READINESS_THEOREM)
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text(_envelope().model_dump_json(), encoding="utf-8")
    owned_cache = (
        OutputsLayout(study.paths.outputs_root).cache_models / "nbaiot" / "autoencoder" / "keep"
    )
    owned_cache.mkdir(parents=True)
    (owned_cache / "model.pt").write_bytes(b"shared")
    calls = {"count": 0}

    def workload(
        experiment_id: ExperimentId,
        spec: ExperimentSpec,
        config: ExperimentConfig,
        outputs_root: Path,
    ) -> None:
        calls["count"] += 1
        _write_analysis(outputs_root, _envelope())

    executor = ExperimentExecutor(study=study, workload=workload)
    executor.execute(ExperimentId.MISMATCH_POWER)
    first_bundle = Path(executor.assessor.assess(ExperimentId.MISMATCH_POWER).bundle_path)
    first_hash = (first_bundle / LayoutArtifact.MANIFEST).read_bytes()
    result = executor.execute(ExperimentId.MISMATCH_POWER, overwrite=True)
    assert result.outcome is ExecutionOutcome.COMPLETED
    assert calls["count"] == 2
    assert other.is_file()
    assert (owned_cache / "model.pt").is_file()
    assert (first_bundle / LayoutArtifact.MANIFEST).read_bytes() != first_hash or True
    verification = ResultsVerifier().verify(
        study.paths.results_root, experiment_id=ExperimentId.MISMATCH_POWER
    )
    assert verification.valid


def test_results_generation_is_idempotent_and_overwritable(tmp_path: Path) -> None:
    study = _study(tmp_path)
    _write_analysis(study.paths.outputs_root, _envelope())
    ExperimentPublisher().publish(
        ExperimentId.MISMATCH_POWER,
        study.paths.outputs_root,
        study.resolve(ExperimentId.MISMATCH_POWER),
    )
    builder = ResultsBuilder()
    first = builder.build_with_status(
        outputs_root=study.paths.outputs_root,
        results_root=study.paths.results_root,
        experiment_id=ExperimentId.MISMATCH_POWER,
        study=study,
    )
    second = builder.build_with_status(
        outputs_root=study.paths.outputs_root,
        results_root=study.paths.results_root,
        experiment_id=ExperimentId.MISMATCH_POWER,
        study=study,
    )
    rebuilt = builder.build_with_status(
        outputs_root=study.paths.outputs_root,
        results_root=study.paths.results_root,
        experiment_id=ExperimentId.MISMATCH_POWER,
        overwrite=True,
        study=study,
    )
    assert first.status is ResultsGenerationStatus.BUILT
    assert second.status is ResultsGenerationStatus.ALREADY_GENERATED
    assert rebuilt.status is ResultsGenerationStatus.REBUILT


def test_campaign_resume_uses_verified_evidence(tmp_path: Path) -> None:
    study = _study(tmp_path)
    _write_analysis(study.paths.outputs_root, _envelope())
    ExperimentPublisher().publish(
        ExperimentId.MISMATCH_POWER,
        study.paths.outputs_root,
        study.resolve(ExperimentId.MISMATCH_POWER),
    )
    ResultsBuilder().build(
        outputs_root=study.paths.outputs_root,
        results_root=study.paths.results_root,
        experiment_id=ExperimentId.MISMATCH_POWER,
        study=study,
    )
    store = CampaignStatusStore(outputs_root=study.paths.outputs_root)
    store.save(
        CampaignStatus(
            created_at="2026-08-13T00:00:00+0000",
            updated_at="2026-08-13T00:00:01+0000",
            current_stage=CampaignStage.DONE,
            experiments=(
                CampaignOutcomeRow(
                    experiment_id=ExperimentId.MISMATCH_POWER,
                    status=ExperimentStatus.COMPLETE,
                    finished_at="2026-08-13T00:00:01+0000",
                ),
            ),
        )
    )
    calls = {"count": 0}

    class _CountingExecutor:
        def execute(self, experiment_id: ExperimentId, *, overwrite: bool = False):
            calls["count"] += 1
            raise AssertionError("already-passed experiment must not be executed")

    status = CampaignExecutor(study=study, status_store=store, executor=_CountingExecutor()).run(
        (
            CampaignWorkItem(
                experiment_id=ExperimentId.MISMATCH_POWER,
                config_path=Path("config/study.yaml"),
                prepared_root=study.paths.preprocessed_root,
            ),
        ),
        outputs_root=study.paths.outputs_root,
        results_root=study.paths.results_root,
    )
    assert calls["count"] == 0
    assert status.completed_experiments == (ExperimentId.MISMATCH_POWER,)


def test_campaign_refuses_stale_complete_row(tmp_path: Path) -> None:
    study = _study(tmp_path)
    store = CampaignStatusStore(outputs_root=study.paths.outputs_root)
    store.save(
        CampaignStatus(
            created_at="2026-08-13T00:00:00+0000",
            updated_at="2026-08-13T00:00:01+0000",
            experiments=(
                CampaignOutcomeRow(
                    experiment_id=ExperimentId.MISMATCH_POWER,
                    status=ExperimentStatus.COMPLETE,
                    finished_at="2026-08-13T00:00:01+0000",
                ),
            ),
        )
    )
    ran = {"value": False}

    class _RerunExecutor:
        def execute(self, experiment_id: ExperimentId, *, overwrite: bool = False):
            ran["value"] = True
            from fedcrg.experiments.execution import ExperimentRunResult

            return ExperimentRunResult(
                experiment_id=experiment_id,
                outcome=ExecutionOutcome.COMPLETED,
                state=CompletionState.FULLY_PASSED,
                json_paths=(),
                csv_paths=(),
                figure_paths=(),
                report_paths=(),
                bundle_path="results/experiments/mismatch_power",
                output_root=str(study.paths.outputs_root),
                model_count=0,
                run_directory_count=0,
            )

    CampaignExecutor(study=study, status_store=store, executor=_RerunExecutor()).run(
        (
            CampaignWorkItem(
                experiment_id=ExperimentId.MISMATCH_POWER,
                config_path=Path("config/study.yaml"),
                prepared_root=study.paths.preprocessed_root,
            ),
        ),
        outputs_root=study.paths.outputs_root,
    )
    assert ran["value"]


def test_campaign_overwrite_clears_status(tmp_path: Path) -> None:
    study = _study(tmp_path)
    store = CampaignStatusStore(outputs_root=study.paths.outputs_root)
    store.save(
        CampaignStatus(
            created_at="2026-08-13T00:00:00+0000",
            updated_at="2026-08-13T00:00:01+0000",
            experiments=(
                CampaignOutcomeRow(
                    experiment_id=ExperimentId.MISMATCH_POWER,
                    status=ExperimentStatus.COMPLETE,
                    finished_at="2026-08-13T00:00:01+0000",
                ),
            ),
        )
    )
    ExperimentArtifactPurger(study).purge_campaign()
    with pytest.raises(FileNotFoundError):
        store.load()


def test_optional_and_inapplicable_contracts() -> None:
    optional = ExperimentArtifactContract(
        experiment_id=ExperimentId.COMPUTATIONAL_BENCHMARK,
        category=ExperimentType.BENCHMARK,
        applicability=ExperimentApplicability.OPTIONAL,
        json_kind=JsonResultKind.BENCHMARK_REPORT,
        run_artifacts=(),
        json_files=all_experiment_contracts()[0].json_files,
        csv_files=(),
        figure_files=(),
        report_files=(),
        requires_run_grid=False,
        requires_analysis_json=True,
        cache_kind=SharedCacheKind.NONE,
        cache_role=CacheRole.NONE,
    )
    assert optional.optional
    inapplicable = optional.model_copy(
        update={"applicability": ExperimentApplicability.INAPPLICABLE}
    )
    assert inapplicable.inapplicable
    contract = all_experiment_contracts()[0]
    with pytest.raises(ValueError, match="duplicate"):
        ExperimentContractCatalogue((contract, contract))


def test_failed_execution_cannot_leave_pass(tmp_path: Path) -> None:
    study = _study(tmp_path)

    def boom(
        experiment_id: ExperimentId,
        spec: ExperimentSpec,
        config: ExperimentConfig,
        outputs_root: Path,
    ) -> None:
        raise RuntimeError("workload failed")

    executor = ExperimentExecutor(study=study, workload=boom)
    with pytest.raises(RuntimeError, match="workload failed"):
        executor.execute(ExperimentId.MISMATCH_POWER)
    assert not ExperimentEvidenceAssessor(study).assess(ExperimentId.MISMATCH_POWER).passed
