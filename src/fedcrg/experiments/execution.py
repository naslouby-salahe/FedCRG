"""Experiment-level execution, idempotent reruns, and publication completion."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from fedcrg.config import ExperimentConfig, ExperimentSpec, Study
from fedcrg.evidence.completion import (
    CompletionAssessment,
    ExperimentArtifactPurger,
    ExperimentEvidenceAssessor,
)
from fedcrg.evidence.contracts import experiment_contract
from fedcrg.experiments.analyses import (
    ProtocolTablePrecomputer,
    RunBenchmark,
    RunSyntheticExperiments,
)
from fedcrg.experiments.runner import RunAllExperiments
from fedcrg.paths import OutputsLayout
from fedcrg.reporting import ExperimentPublisher, ResultsBuilder
from fedcrg.types import (
    CompletionState,
    ExecutionOutcome,
    ExperimentId,
    NonNegativeCount,
    PathString,
)

Frozen = ConfigDict(frozen=True, extra="forbid")


class ExperimentRunResult(BaseModel):
    """Outcome of `fedcrg run` after evidence assessment and optional execution."""

    model_config = Frozen

    experiment_id: ExperimentId
    outcome: ExecutionOutcome
    state: CompletionState
    json_paths: tuple[PathString, ...]
    csv_paths: tuple[PathString, ...]
    figure_paths: tuple[PathString, ...]
    report_paths: tuple[PathString, ...]
    bundle_path: PathString
    output_root: PathString
    model_count: NonNegativeCount
    run_directory_count: NonNegativeCount


WorkloadRunner = Callable[[ExperimentId, ExperimentSpec, ExperimentConfig, Path], None]


class ExperimentExecutor:
    """Runs one experiment through execution, publication, bundle rebuild, and verification."""

    def __init__(
        self,
        study: Study | None = None,
        assessor: ExperimentEvidenceAssessor | None = None,
        purger: ExperimentArtifactPurger | None = None,
        publisher: ExperimentPublisher | None = None,
        results: ResultsBuilder | None = None,
        empirical: RunAllExperiments | None = None,
        workload: WorkloadRunner | None = None,
    ) -> None:
        self.study = study or Study.load()
        self.assessor = assessor or ExperimentEvidenceAssessor(self.study)
        self.purger = purger or ExperimentArtifactPurger(self.study)
        self.publisher = publisher or ExperimentPublisher()
        self.results = results or ResultsBuilder()
        self.empirical = empirical or RunAllExperiments()
        self.workload = workload

    def execute(
        self, experiment_id: ExperimentId, *, overwrite: bool = False
    ) -> ExperimentRunResult:
        """Validate existing evidence, skip if already passed, or run and publish to a verified pass."""
        if overwrite:
            self.purger.purge(experiment_id)
        assessment = self.assessor.assess(experiment_id)
        if assessment.passed:
            return self._result(ExecutionOutcome.ALREADY_PASSED, assessment, model_count=0)
        spec = self.study.spec(experiment_id)
        config = self.study.resolve(experiment_id)
        outputs_root = config.outputs_root
        model_count = 0
        run_count = assessment.complete_run_count
        if assessment.state in {
            CompletionState.NOT_STARTED,
            CompletionState.EXECUTION_INCOMPLETE,
            CompletionState.FAILED,
            CompletionState.INVALID,
        }:
            workload = self._run_workload(experiment_id, spec, config, outputs_root)
            model_count = workload[0]
            run_count = workload[1]
        self.publisher.publish(experiment_id, outputs_root, config)
        self.results.build(
            outputs_root=outputs_root,
            results_root=self.study.paths.results_root,
            experiment_id=experiment_id,
            overwrite=True,
            study=self.study,
        )
        final = self.assessor.assess(experiment_id)
        if not final.passed:
            raise RuntimeError(
                "Experiment evidence is not fully passed after execution: "
                + ", ".join(item.detail for item in final.problems)
            )
        return self._result(
            ExecutionOutcome.COMPLETED, final, model_count=model_count, run_count=run_count
        )

    def _run_workload(
        self,
        experiment_id: ExperimentId,
        spec: ExperimentSpec,
        config: ExperimentConfig,
        outputs_root: Path,
    ) -> tuple[NonNegativeCount, NonNegativeCount]:
        if self.workload is not None:
            self.workload(experiment_id, spec, config, outputs_root)
            return 0, 0
        ProtocolTablePrecomputer().precompute(config, spec)
        contract = experiment_contract(experiment_id)
        if contract.requires_analysis_json:
            output = OutputsLayout(outputs_root).analysis_result(experiment_id)
            if experiment_id is ExperimentId.COMPUTATIONAL_BENCHMARK:
                RunBenchmark(spec, config).run(output)
            else:
                RunSyntheticExperiments().run(experiment_id, spec, config, output)
            return 0, 0
        executed = self.empirical.execute(experiment_id, config, config.preprocessed_root)
        return len(executed.models), len(executed.run_directories)

    @staticmethod
    def _result(
        outcome: ExecutionOutcome,
        assessment: CompletionAssessment,
        *,
        model_count: NonNegativeCount,
        run_count: NonNegativeCount | None = None,
    ) -> ExperimentRunResult:
        return ExperimentRunResult(
            experiment_id=assessment.experiment_id,
            outcome=outcome,
            state=assessment.state,
            json_paths=assessment.json_paths,
            csv_paths=assessment.csv_paths,
            figure_paths=assessment.figure_paths,
            report_paths=assessment.report_paths,
            bundle_path=assessment.bundle_path,
            output_root=assessment.output_root,
            model_count=model_count,
            run_directory_count=assessment.complete_run_count if run_count is None else run_count,
        )
