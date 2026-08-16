"""Single experiment-level completeness assessor and owned-artifact purge."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pydantic
from pydantic import BaseModel, ConfigDict

from fedcrg.config import ExperimentConfig, Study
from fedcrg.evidence.contracts import (
    ExperimentArtifactContract,
    RequiredResultFile,
    detector_cache_identity,
    experiment_contract,
)
from fedcrg.evidence.models import (
    ChecksumRecord,
    EmpiricalExperimentResult,
    ExperimentProvenance,
    RunConfig,
    RunManifest,
)
from fedcrg.experiments.analyses import BenchmarkReport, SyntheticExperimentEnvelope
from fedcrg.hashing import sha256_file
from fedcrg.paths import (
    ExperimentResultsBundleLayout,
    LayoutArtifact,
    OutputsLayout,
    ResultsBundleLayout,
    RunLayout,
    experiment_results_root,
)
from fedcrg.types import (
    CacheRole,
    CompletionState,
    Description,
    ExperimentId,
    ExperimentStatus,
    JsonResultKind,
    NonNegativeCount,
    PathString,
    ResultFormat,
    Sha256,
)

Frozen = ConfigDict(frozen=True, extra="forbid")


class CompletionProblem(BaseModel):
    """One completeness or verification problem for an experiment."""

    model_config = Frozen

    code: Description
    detail: Description


class CompletionAssessment(BaseModel):
    """Derived experiment-level completion state and the problems that produced it."""

    model_config = Frozen

    experiment_id: ExperimentId
    state: CompletionState
    problems: tuple[CompletionProblem, ...]
    config_hash: Sha256
    json_paths: tuple[PathString, ...]
    csv_paths: tuple[PathString, ...]
    figure_paths: tuple[PathString, ...]
    report_paths: tuple[PathString, ...]
    bundle_path: PathString
    output_root: PathString
    complete_run_count: NonNegativeCount
    expected_run_count: NonNegativeCount

    @property
    def passed(self) -> bool:
        return self.state is CompletionState.FULLY_PASSED


def _problem(code: Description, detail: Description) -> CompletionProblem:
    return CompletionProblem(code=code, detail=detail)


def expected_run_count(config: ExperimentConfig) -> NonNegativeCount:
    """Seed-by-policy grid size for an empirical experiment."""
    return (
        len(config.randomness.model_seeds)
        * len(config.dataset.calibration_seeds)
        * len(config.policies)
    )


def experiment_source_files(
    contract: ExperimentArtifactContract,
    outputs_root: Path,
    experiment_id: ExperimentId,
) -> tuple[Path, ...]:
    """Source artifacts that must be current for the experiment result bundle."""
    outputs = OutputsLayout(outputs_root)
    paths: list[Path] = []
    analysis = outputs.analysis_result(experiment_id)
    if contract.requires_analysis_json:
        paths.append(analysis)
    published = (
        *tuple(
            outputs.experiment_tables(experiment_id) / item.filename for item in contract.csv_files
        ),
        *tuple(
            outputs.experiment_figures(experiment_id) / item.filename
            for item in contract.figure_files
        ),
        *tuple(
            outputs.experiment_reports(experiment_id) / item.filename
            for item in contract.report_files
        ),
        outputs.experiment_reports(experiment_id) / LayoutArtifact.EXPERIMENT_RESULT_JSON,
    )
    paths.extend(path for path in published if path.is_file())
    return tuple(paths)


def source_digests(
    contract: ExperimentArtifactContract,
    outputs_root: Path,
    experiment_id: ExperimentId,
) -> tuple[ChecksumRecord, ...]:
    """Hashes of current source artifacts used to detect a stale delivery bundle."""
    records: list[ChecksumRecord] = []
    for path in experiment_source_files(contract, outputs_root, experiment_id):
        if not path.is_file():
            continue
        records.append(
            ChecksumRecord(
                relative_path=path.as_posix(),
                sha256=sha256_file(path),
            )
        )
    return tuple(records)


class ExperimentEvidenceAssessor:
    """Derives experiment completion from the artifact contract and verified files."""

    def __init__(self, study: Study | None = None) -> None:
        self.study = study or Study.load()

    def assess(
        self,
        experiment_id: ExperimentId,
        *,
        outputs_root: Path | None = None,
        results_root: Path | None = None,
    ) -> CompletionAssessment:
        """Evaluate one experiment's execution, publication, and delivery evidence."""
        contract = experiment_contract(experiment_id)
        config = self.study.resolve(experiment_id)
        outputs_root = outputs_root or config.outputs_root
        results_root = results_root or self.study.paths.results_root
        outputs = OutputsLayout(outputs_root)
        bundle_root = experiment_results_root(results_root, experiment_id)
        expected = expected_run_count(config) if contract.requires_run_grid else 0
        runs = self._completed_runs(experiment_id, outputs, config.config_hash)
        problems: list[CompletionProblem] = []

        if contract.inapplicable:
            return self._assessment(
                experiment_id,
                CompletionState.FULLY_PASSED,
                (),
                config,
                bundle_root,
                complete_run_count=len(runs),
                expected_run_count=expected,
            )

        execution_state, execution_problems = self._execution_state(
            contract, experiment_id, outputs, runs, expected
        )
        problems.extend(execution_problems)

        publication_problems = self._publication_problems(contract, experiment_id, outputs)
        problems.extend(publication_problems)

        bundle_state, bundle_problems = self._bundle_state(
            contract, experiment_id, config, outputs_root, results_root, bundle_root
        )
        problems.extend(bundle_problems)

        state = self._combine(
            contract,
            execution_state,
            publication_problems,
            bundle_state,
            outputs,
            experiment_id,
        )
        return self._assessment(
            experiment_id,
            state,
            tuple(problems),
            config,
            bundle_root,
            complete_run_count=len(runs),
            expected_run_count=expected,
        )

    def _assessment(
        self,
        experiment_id: ExperimentId,
        state: CompletionState,
        problems: tuple[CompletionProblem, ...],
        config: ExperimentConfig,
        bundle_root: Path,
        *,
        complete_run_count: NonNegativeCount,
        expected_run_count: NonNegativeCount,
    ) -> CompletionAssessment:
        contract = experiment_contract(experiment_id)
        outputs = OutputsLayout(config.outputs_root)
        return CompletionAssessment(
            experiment_id=experiment_id,
            state=state,
            problems=problems,
            config_hash=config.config_hash,
            json_paths=self._existing(
                (
                    outputs.experiment_reports(experiment_id)
                    / LayoutArtifact.EXPERIMENT_RESULT_JSON,
                    *tuple(
                        ExperimentResultsBundleLayout(bundle_root).json_dir / item.filename
                        for item in contract.json_files
                    ),
                )
            ),
            csv_paths=self._existing(
                tuple(
                    outputs.experiment_tables(experiment_id) / item.filename
                    for item in contract.csv_files
                )
            ),
            figure_paths=self._existing(
                tuple(
                    outputs.experiment_figures(experiment_id) / item.filename
                    for item in contract.figure_files
                )
            ),
            report_paths=self._existing(
                tuple(
                    outputs.experiment_reports(experiment_id) / item.filename
                    for item in contract.report_files
                )
            ),
            bundle_path=bundle_root.as_posix(),
            output_root=config.outputs_root.as_posix(),
            complete_run_count=complete_run_count,
            expected_run_count=expected_run_count,
        )

    @staticmethod
    def _existing(paths: tuple[Path, ...]) -> tuple[PathString, ...]:
        return tuple(path.as_posix() for path in paths if path.is_file())

    def _execution_state(
        self,
        contract: ExperimentArtifactContract,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
        runs: tuple[Path, ...],
        expected: NonNegativeCount,
    ) -> tuple[CompletionState, tuple[CompletionProblem, ...]]:
        problems: list[CompletionProblem] = []
        if self._has_running(experiment_id, outputs):
            return CompletionState.RUNNING, (
                _problem("running", "A run directory is still in a non-terminal state"),
            )
        if self._has_failed(experiment_id, outputs) and (
            not contract.requires_run_grid or len(runs) < expected
        ):
            problems.append(_problem("failed_run", "A run directory recorded FAILED"))
            return CompletionState.FAILED, tuple(problems)

        if contract.requires_analysis_json:
            return self._analysis_execution_state(contract, experiment_id, outputs)

        if expected == 0:
            return CompletionState.NOT_STARTED, (
                _problem("empty_grid", "Empirical experiment has an empty seed/policy grid"),
            )
        if not runs and not self._has_any_run(experiment_id, outputs):
            return CompletionState.NOT_STARTED, (
                _problem("not_started", "No run directories exist for this experiment"),
            )
        if len(runs) != expected:
            problems.append(
                _problem(
                    "run_grid",
                    f"Completed verified runs {len(runs)} != expected {expected}",
                )
            )
            return CompletionState.EXECUTION_INCOMPLETE, tuple(problems)
        return CompletionState.ANALYSIS_COMPLETE, ()

    def _analysis_execution_state(
        self,
        contract: ExperimentArtifactContract,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
    ) -> tuple[CompletionState, tuple[CompletionProblem, ...]]:
        analysis = outputs.analysis_result(experiment_id)
        analysis_problems = self._validate_json_kind(contract, analysis)
        if analysis_problems:
            if not analysis.exists():
                if self._any_experiment_artifact(contract, experiment_id, outputs):
                    return CompletionState.EXECUTION_INCOMPLETE, tuple(analysis_problems)
                return CompletionState.NOT_STARTED, tuple(analysis_problems)
            return CompletionState.INVALID, tuple(analysis_problems)
        return CompletionState.ANALYSIS_COMPLETE, ()

    def _publication_problems(
        self,
        contract: ExperimentArtifactContract,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
    ) -> tuple[CompletionProblem, ...]:
        if contract.optional or contract.inapplicable:
            return ()
        problems: list[CompletionProblem] = []
        result_json = (
            outputs.experiment_reports(experiment_id) / LayoutArtifact.EXPERIMENT_RESULT_JSON
        )
        problems.extend(self._validate_json_kind(contract, result_json))
        problems.extend(
            self._validate_required_files(
                contract.csv_files,
                outputs.experiment_tables(experiment_id),
                ResultFormat.CSV,
            )
        )
        problems.extend(
            self._validate_required_files(
                contract.figure_files,
                outputs.experiment_figures(experiment_id),
                ResultFormat.FIGURE,
            )
        )
        problems.extend(
            self._validate_required_files(
                contract.report_files,
                outputs.experiment_reports(experiment_id),
                ResultFormat.REPORT,
            )
        )
        return tuple(problems)

    def _bundle_state(
        self,
        contract: ExperimentArtifactContract,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        outputs_root: Path,
        results_root: Path,
        bundle_root: Path,
    ) -> tuple[CompletionState | None, tuple[CompletionProblem, ...]]:
        layout = ExperimentResultsBundleLayout(bundle_root)
        if not layout.manifest.is_file():
            return None, (
                _problem("missing_bundle", f"Experiment result bundle missing: {bundle_root}"),
            )
        from fedcrg.reporting import ResultsVerifier

        verification = ResultsVerifier().verify(results_root, experiment_id=experiment_id)
        problems = tuple(_problem("bundle", item) for item in verification.problems)
        if not verification.valid:
            return CompletionState.INVALID, problems
        if not layout.provenance_json.is_file():
            return CompletionState.EVIDENCE_INCOMPLETE, (
                _problem("missing_provenance", "Experiment bundle has no provenance record"),
            )
        try:
            provenance = ExperimentProvenance.model_validate_json(
                layout.provenance_json.read_text(encoding="utf-8")
            )
        except pydantic.ValidationError as exc:
            return CompletionState.INVALID, (_problem("malformed_provenance", str(exc)[:200]),)
        if provenance.config_hash != config.config_hash:
            return CompletionState.STALE, (
                _problem(
                    "stale_config",
                    "Bundle config hash does not match the current experiment configuration",
                ),
            )
        current = {
            item.relative_path: item.sha256
            for item in source_digests(contract, outputs_root, experiment_id)
        }
        recorded = {item.relative_path: item.sha256 for item in provenance.source_digests}
        if current != recorded:
            return CompletionState.STALE, (
                _problem(
                    "stale_bundle",
                    "Bundle source digests do not match current execution artifacts",
                ),
            )
        return CompletionState.FULLY_PASSED, ()

    @staticmethod
    def _combine(
        contract: ExperimentArtifactContract,
        execution_state: CompletionState,
        publication_problems: tuple[CompletionProblem, ...],
        bundle_state: CompletionState | None,
        outputs: OutputsLayout,
        experiment_id: ExperimentId,
    ) -> CompletionState:
        if execution_state in {
            CompletionState.NOT_STARTED,
            CompletionState.RUNNING,
            CompletionState.FAILED,
            CompletionState.INVALID,
            CompletionState.EXECUTION_INCOMPLETE,
        }:
            return execution_state
        if publication_problems:
            return CompletionState.EVIDENCE_INCOMPLETE
        if bundle_state is CompletionState.STALE:
            return CompletionState.STALE
        if bundle_state is CompletionState.INVALID:
            return CompletionState.INVALID
        if bundle_state is CompletionState.FULLY_PASSED:
            return CompletionState.FULLY_PASSED
        if ExperimentEvidenceAssessor._any_experiment_artifact(contract, experiment_id, outputs):
            return CompletionState.EVIDENCE_INCOMPLETE
        return CompletionState.EVIDENCE_INCOMPLETE

    def _validate_json_kind(
        self, contract: ExperimentArtifactContract, path: Path
    ) -> list[CompletionProblem]:
        problem = self._json_kind_problem(contract, path)
        return [problem] if problem else []

    def _json_kind_problem(
        self, contract: ExperimentArtifactContract, path: Path
    ) -> CompletionProblem | None:
        if not path.is_file():
            return _problem("missing_json", f"Missing JSON result: {path.name}")
        if path.stat().st_size == 0:
            return _problem("empty_json", f"Empty JSON result: {path.name}")
        try:
            payload = path.read_text(encoding="utf-8")
        except OSError as exc:
            return _problem("unreadable_json", str(exc)[:200])
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return _problem("malformed_json", f"JSON does not parse: {path.name}")
        if not isinstance(parsed, dict) or not parsed:
            return _problem("empty_json", f"JSON result has no content: {path.name}")
        try:
            if contract.json_kind is JsonResultKind.SYNTHETIC_ENVELOPE:
                envelope = SyntheticExperimentEnvelope.model_validate_json(payload)
                if not envelope.cells:
                    return _problem("empty_json", "Synthetic envelope contains no cells")
            elif contract.json_kind is JsonResultKind.BENCHMARK_REPORT:
                report = BenchmarkReport.model_validate_json(payload)
                if not report.cells:
                    return _problem("empty_json", "Benchmark report contains no cells")
            else:
                result = EmpiricalExperimentResult.model_validate_json(payload)
                if not result.records:
                    return _problem("empty_json", "Empirical result contains no records")
        except pydantic.ValidationError:
            return _problem("malformed_json", f"JSON failed its typed schema: {path.name}")
        return None

    @staticmethod
    def _validate_required_files(
        files: tuple[RequiredResultFile, ...],
        directory: Path,
        kind: ResultFormat,
    ) -> tuple[CompletionProblem, ...]:
        problems: list[CompletionProblem] = []
        for item in files:
            path = directory / item.filename
            label = f"{kind}:{item.filename}"
            if not path.is_file():
                problems.append(_problem(f"missing_{kind}", f"Missing {label}"))
                continue
            if path.stat().st_size == 0:
                problems.append(_problem(f"empty_{kind}", f"Empty {label}"))
                continue
            if kind is ResultFormat.CSV:
                try:
                    frame = pd.read_csv(path)
                except (OSError, pd.errors.ParserError, UnicodeError):
                    problems.append(_problem("malformed_csv", f"Unreadable {label}"))
                    continue
                if frame.empty:
                    problems.append(_problem("empty_csv", f"CSV has no data rows: {label}"))
        return tuple(problems)

    def _completed_runs(
        self, experiment_id: ExperimentId, outputs: OutputsLayout, config_hash: Sha256
    ) -> tuple[Path, ...]:
        if not outputs.runs.is_dir():
            return ()
        rows: list[Path] = []
        for run_dir in sorted(path for path in outputs.runs.iterdir() if path.is_dir()):
            manifest = self._run_manifest(run_dir)
            if manifest is None:
                continue
            if manifest.experiment_id is not experiment_id:
                continue
            if manifest.status is not ExperimentStatus.COMPLETE:
                continue
            if manifest.config_hash != config_hash:
                continue
            rows.append(run_dir)
        return tuple(rows)

    def _has_any_run(self, experiment_id: ExperimentId, outputs: OutputsLayout) -> bool:
        if not outputs.runs.is_dir():
            return False
        for run_dir in outputs.runs.iterdir():
            if not run_dir.is_dir():
                continue
            manifest = self._run_manifest(run_dir)
            if manifest is not None and manifest.experiment_id is experiment_id:
                return True
        return False

    def _has_running(self, experiment_id: ExperimentId, outputs: OutputsLayout) -> bool:
        if not outputs.runs.is_dir():
            return False
        terminal = {ExperimentStatus.COMPLETE, ExperimentStatus.FAILED, ExperimentStatus.BLOCKED}
        for run_dir in outputs.runs.iterdir():
            if not run_dir.is_dir():
                continue
            manifest = self._run_manifest(run_dir)
            if manifest is None or manifest.experiment_id is not experiment_id:
                continue
            if manifest.status not in terminal:
                return True
        return False

    def _has_failed(self, experiment_id: ExperimentId, outputs: OutputsLayout) -> bool:
        if not outputs.runs.is_dir():
            return False
        for run_dir in outputs.runs.iterdir():
            if not run_dir.is_dir():
                continue
            manifest = self._run_manifest(run_dir)
            if (
                manifest is not None
                and manifest.experiment_id is experiment_id
                and manifest.status is ExperimentStatus.FAILED
            ):
                return True
        return False

    @staticmethod
    def _any_experiment_artifact(
        contract: ExperimentArtifactContract,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
    ) -> bool:
        candidates = (
            outputs.analysis_result(experiment_id),
            outputs.experiment_reports(experiment_id) / LayoutArtifact.EXPERIMENT_RESULT_JSON,
            *(
                outputs.experiment_tables(experiment_id) / item.filename
                for item in contract.csv_files
            ),
            *(
                outputs.experiment_figures(experiment_id) / item.filename
                for item in contract.figure_files
            ),
            *(
                outputs.experiment_reports(experiment_id) / item.filename
                for item in contract.report_files
            ),
        )
        return any(path.exists() for path in candidates)

    @staticmethod
    def _run_manifest(run_dir: Path) -> RunManifest | None:
        path = RunLayout(run_dir).manifest
        if not path.is_file():
            return None
        try:
            return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, pydantic.ValidationError):
            return None


class ExperimentArtifactPurger:
    """Deletes artifacts owned by an experiment without touching shared prerequisites."""

    def __init__(self, study: Study | None = None) -> None:
        self.study = study or Study.load()

    def purge(self, experiment_id: ExperimentId) -> None:
        """Remove run, analysis, publication, cache, and bundle artifacts this experiment owns."""
        contract = experiment_contract(experiment_id)
        outputs = OutputsLayout(self.study.paths.outputs_root)
        self._purge_runs(experiment_id, outputs)
        analysis = outputs.analysis_result(experiment_id)
        if analysis.is_file():
            analysis.unlink()
        self._rmtree(outputs.experiment_reports(experiment_id))
        self._rmtree(outputs.experiment_tables(experiment_id))
        self._rmtree(outputs.experiment_figures(experiment_id))
        self._rmtree(experiment_results_root(self.study.paths.results_root, experiment_id))
        if contract.cache_role is CacheRole.OWNER:
            self._purge_owned_cache(contract, outputs)
        self._drop_campaign_row(experiment_id, outputs)

    def purge_campaign(self) -> None:
        """Clear campaign-owned state and every experiment-owned regenerable artifact."""
        outputs = OutputsLayout(self.study.paths.outputs_root)
        status = outputs.campaign_status()
        if status.is_file():
            status.unlink()
        self._purge_campaign_bundle(self.study.paths.results_root)
        for contract in (experiment_contract(item.id) for item in self.study.catalogue.all()):
            self.purge(contract.experiment_id)

    def _purge_runs(self, experiment_id: ExperimentId, outputs: OutputsLayout) -> None:
        if not outputs.runs.is_dir():
            return
        for run_dir in tuple(outputs.runs.iterdir()):
            if not run_dir.is_dir():
                continue
            if self._run_belongs(run_dir, experiment_id):
                self._rmtree(run_dir)

    @staticmethod
    def _run_belongs(run_dir: Path, experiment_id: ExperimentId) -> bool:
        layout = RunLayout(run_dir)
        if layout.run_config.is_file():
            try:
                config = RunConfig.model_validate_json(
                    layout.run_config.read_text(encoding="utf-8")
                )
            except (OSError, pydantic.ValidationError):
                config = None
            if config is not None:
                return config.experiment_id is experiment_id
        if layout.manifest.is_file():
            try:
                manifest = RunManifest.model_validate_json(
                    layout.manifest.read_text(encoding="utf-8")
                )
            except (OSError, pydantic.ValidationError):
                return False
            return manifest.experiment_id is experiment_id
        return False

    @staticmethod
    def _purge_owned_cache(contract: ExperimentArtifactContract, outputs: OutputsLayout) -> None:
        identity = detector_cache_identity(contract.cache_kind)
        if identity is None:
            return
        for root in (outputs.cache_models, outputs.cache_scores):
            target = root / identity.dataset_id / identity.detector_id
            if target.is_dir():
                shutil.rmtree(target)

    def _drop_campaign_row(self, experiment_id: ExperimentId, outputs: OutputsLayout) -> None:
        from fedcrg.experiments.runner import CampaignStatus, CampaignStatusStore

        store = CampaignStatusStore(outputs_root=outputs.outputs_root)
        try:
            status = store.load()
        except FileNotFoundError:
            return
        remaining = tuple(
            row for row in status.experiments if row.experiment_id is not experiment_id
        )
        if remaining == status.experiments:
            return
        store.save(
            CampaignStatus(
                created_at=status.created_at,
                updated_at=status.updated_at,
                current_experiment=None
                if status.current_experiment is experiment_id
                else status.current_experiment,
                current_stage=status.current_stage,
                experiments=remaining,
                results_path=status.results_path,
                elapsed_seconds=status.elapsed_seconds,
            )
        )

    @staticmethod
    def _purge_campaign_bundle(results_root: Path) -> None:
        layout = ResultsBundleLayout(results_root)
        for path in (
            layout.manifest,
            layout.checksums,
        ):
            if path.is_file():
                path.unlink()
        for directory in layout.required_directories:
            if directory.is_dir():
                shutil.rmtree(directory)

    @staticmethod
    def _rmtree(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
