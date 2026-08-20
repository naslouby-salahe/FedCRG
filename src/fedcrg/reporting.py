from __future__ import annotations

import json
import os

import pydantic
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fedcrg.config import ExperimentConfig, Study
from fedcrg.evidence.contracts import ExperimentArtifactContract
from fedcrg.evidence.models import (
    ChecksumRecord,
    EmpiricalExperimentResult,
    EmpiricalPolicyResult,
    ExperimentProvenance,
    MetricRecord,
    RunManifest,
)
from fedcrg.evidence.store import (
    atomic_write_json,
)
from fedcrg.hashing import sha256_file
from fedcrg.paths import (
    ExperimentResultsBundleLayout,
    LayoutArtifact,
    OutputsLayout,
    RunLayout,
    experiment_results_root,
)
from fedcrg.experiments.analyses import (
    BenchmarkReport,
    SyntheticExperimentEnvelope,
)
from fedcrg.thresholding.metrics import FederationMetrics
from fedcrg.types import (
    CalibrationSeed,
    ClientId,
    Description,
    DetectorId,
    CompletionState,
    DatasetId,
    ExperimentId,
    ExperimentStatus,
    JsonResultKind,
    ResultsGenerationStatus,
    Fpr,
    Identifier,
    ModelSeed,
    PathString,
    PolicyId,
    PositiveCount,
    Sha256,
)
from pydantic import BaseModel, ConfigDict

Frozen = ConfigDict(frozen=True)


def _run_manifest(run_dir: Path) -> RunManifest | None:
    manifest_path = RunLayout(run_dir).manifest
    if not manifest_path.is_file():
        return None
    try:
        return RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except pydantic.ValidationError:
        return None


def _band_guide_lines() -> tuple[Fpr, Fpr, Fpr]:
    protocol = Study.load().study_config.protocol
    return protocol.band.lower, protocol.alpha, protocol.band.upper


def build_per_client_operating_points_figure(output: Path, frame: pd.DataFrame) -> Path:
    if frame.empty:
        raise ValueError("operating-points figure requires a non-empty results table")
    if "client_id" not in frame.columns or "fpr" not in frame.columns:
        raise ValueError("primary policy results table lacks client_id/fpr columns")
    figure, axis = plt.subplots(figsize=(9, 5))
    for policy, group in frame.groupby("policy_id", sort=True):
        axis.scatter(group["client_id"], group["fpr"], label=policy, s=18)
    for level in _band_guide_lines():
        axis.axhline(level, color="gray", linestyle=":", linewidth=0.8)
    axis.set_xlabel("client id")
    axis.set_ylabel("final-test FPR")
    axis.set_title("Per-client operating points")
    axis.legend(fontsize=8, ncol=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def build_reliability_utility_frontier_figure(output: Path, frame: pd.DataFrame) -> Path:
    if frame.empty:
        raise ValueError("reliability-utility figure requires a non-empty results table")
    required = {"policy_id", "mebe", "attack_balanced_macro_tpr"}
    if not required.issubset(frame.columns):
        raise ValueError("primary policy results table lacks MEBE/ABMacroTPR columns")
    summary = frame.groupby("policy_id", sort=True)[["mebe", "attack_balanced_macro_tpr"]].mean()
    summary = pd.DataFrame(summary).reset_index()
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(summary["mebe"], summary["attack_balanced_macro_tpr"], s=42)
    for _, row in summary.iterrows():
        policy_label = str(row["policy_id"])
        axis.annotate(policy_label, (row["mebe"], row["attack_balanced_macro_tpr"]), fontsize=8)
    axis.set_xlabel("mean excess band error (MEBE)")
    axis.set_ylabel("attack-balanced macro TPR")
    axis.set_title("Reliability-utility frontier")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def build_external_replication_figure(output: Path, frame: pd.DataFrame) -> Path:
    if frame.empty:
        raise ValueError("external-replication figure requires a non-empty results table")
    if "client_id" not in frame.columns or "fpr" not in frame.columns:
        raise ValueError("external replication table lacks client_id/fpr columns")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(range(len(frame)), frame["fpr"], s=18)
    for level in _band_guide_lines():
        axis.axhline(level, color="gray", linestyle=":", linewidth=0.8)
    axis.set_xlabel("eligible DIAD client")
    axis.set_ylabel("final-test FPR")
    axis.set_title("External replication")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


class OperatingPointRow(BaseModel):
    model_config = Frozen

    client_id: ClientId
    fpr: Fpr
    policy_id: PolicyId


class ExperimentPublication(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    experiment_id: ExperimentId
    json_path: Path
    csv_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]


class ExperimentPublisher:
    def publish(
        self, experiment_id: ExperimentId, outputs_root: Path, config: ExperimentConfig
    ) -> ExperimentPublication:
        from fedcrg.evidence.contracts import experiment_contract

        contract = experiment_contract(experiment_id)
        outputs = OutputsLayout(outputs_root)
        json_path = self._write_json(contract.json_kind, experiment_id, outputs, config)
        csv_paths = self._write_csv(contract, experiment_id, outputs, json_path)
        figure_paths = self._write_figures(contract, experiment_id, outputs, csv_paths)
        return ExperimentPublication(
            experiment_id=experiment_id,
            json_path=json_path,
            csv_paths=csv_paths,
            figure_paths=figure_paths,
        )

    def _write_json(
        self,
        kind: JsonResultKind,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
        config: ExperimentConfig,
    ) -> Path:
        destination = outputs.experiment_json(experiment_id) / LayoutArtifact.EXPERIMENT_RESULT_JSON
        destination.parent.mkdir(parents=True, exist_ok=True)
        if kind is JsonResultKind.EMPIRICAL_RESULTS:
            payload = self._empirical_result(experiment_id, outputs, config)
            atomic_write_json(destination, payload)
            return destination
        source = outputs.analysis_result(experiment_id)
        if not source.is_file():
            raise FileNotFoundError(f"Missing analysis JSON for {experiment_id}: {source}")
        if kind is JsonResultKind.SYNTHETIC_ENVELOPE:
            envelope = SyntheticExperimentEnvelope.model_validate_json(
                source.read_text(encoding="utf-8")
            )
            if not envelope.cells:
                raise ValueError(f"{experiment_id} analysis JSON contains no cells")
            atomic_write_json(destination, envelope)
            return destination
        report = BenchmarkReport.model_validate_json(source.read_text(encoding="utf-8"))
        if not report.cells:
            raise ValueError(f"{experiment_id} benchmark JSON contains no cells")
        atomic_write_json(destination, report)
        return destination

    @staticmethod
    def _empirical_result(
        experiment_id: ExperimentId, outputs: OutputsLayout, config: ExperimentConfig
    ) -> EmpiricalExperimentResult:
        if not outputs.runs.is_dir():
            raise FileNotFoundError(f"No run directories exist for {experiment_id}")
        records: list[EmpiricalPolicyResult] = []
        for run_dir in sorted(path for path in outputs.runs.iterdir() if path.is_dir()):
            layout = RunLayout(run_dir)
            if not layout.manifest.is_file() or not layout.federation_metrics.is_file():
                continue
            try:
                manifest = RunManifest.model_validate_json(
                    layout.manifest.read_text(encoding="utf-8")
                )
            except pydantic.ValidationError:
                continue
            if (
                manifest.experiment_id is not experiment_id
                or manifest.status is not ExperimentStatus.COMPLETE
                or manifest.config_hash != config.config_hash
            ):
                continue
            federation = FederationMetrics.model_validate_json(
                layout.federation_metrics.read_text(encoding="utf-8")
            )
            records.append(
                EmpiricalPolicyResult(
                    run_id=manifest.run_id,
                    policy_id=manifest.policy_id,
                    model_seed=manifest.model_seed,
                    calibration_seed=manifest.calibration_seed,
                    config_hash=manifest.config_hash,
                    mebe=federation.mebe,
                    high_excess=federation.high_excess,
                    band_violation_rate=federation.band_violation_rate,
                    attack_balanced_macro_tpr=federation.attack_balanced_macro_tpr,
                )
            )
        if not records:
            raise ValueError(f"{experiment_id} has no completed federation metrics to publish")
        return EmpiricalExperimentResult(
            experiment_id=experiment_id,
            config_hash=config.config_hash,
            records=tuple(records),
        )

    def _write_csv(
        self,
        contract: ExperimentArtifactContract,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
        json_path: Path,
    ) -> tuple[Path, ...]:
        if not contract.csv_files:
            return ()
        table_root = outputs.experiment_tables(experiment_id)
        table_root.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for item in contract.csv_files:
            destination = table_root / item.filename
            if "cells" in payload:
                frame = pd.DataFrame.from_records(payload["cells"])
            elif "records" in payload:
                frame = pd.DataFrame.from_records(payload["records"])
            else:
                raise ValueError(f"{experiment_id} JSON result is not tabular")
            if frame.empty:
                raise ValueError(f"{experiment_id} produced an empty CSV")
            frame.to_csv(destination, index=False)
            written.append(destination)
        return tuple(written)

    def _write_figures(
        self,
        contract: ExperimentArtifactContract,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
        csv_paths: tuple[Path, ...],
    ) -> tuple[Path, ...]:
        if not contract.figure_files:
            return ()
        figure_root = outputs.experiment_figures(experiment_id)
        figure_root.mkdir(parents=True, exist_ok=True)
        frame = pd.read_csv(csv_paths[0]) if csv_paths else pd.DataFrame()
        written: list[Path] = []
        for item in contract.figure_files:
            destination = figure_root / item.filename
            if item.filename is LayoutArtifact.EXPERIMENT_POWER_FIGURE:
                self._coverage_or_power_figure(destination, frame, power=True)
            elif item.filename is LayoutArtifact.EXPERIMENT_COVERAGE_FIGURE:
                self._coverage_or_power_figure(destination, frame, power=False)
            elif item.filename is LayoutArtifact.EXPERIMENT_OPERATING_POINTS_FIGURE:
                self._operating_points_from_runs(destination, experiment_id, outputs)
            elif item.filename is LayoutArtifact.EXPERIMENT_RELIABILITY_FIGURE:
                build_reliability_utility_frontier_figure(destination, frame)
            elif item.filename is LayoutArtifact.EXPERIMENT_REPLICATION_FIGURE:
                self._replication_figure(destination, experiment_id, outputs, frame)
            else:
                raise ValueError(f"Unsupported figure {item.filename} for {experiment_id}")
            written.append(destination)
        return tuple(written)

    @staticmethod
    def _coverage_or_power_figure(output: Path, frame: pd.DataFrame, *, power: bool) -> Path:
        if frame.empty:
            raise ValueError("Figure requires a non-empty results table")
        figure, axis = plt.subplots(figsize=(8, 5))
        if power and "declaration_probability" in frame.columns:
            if {"sample_count", "true_fpr", "declaration_probability"}.issubset(frame.columns):
                pivot = frame.pivot_table(
                    index="sample_count",
                    columns="true_fpr",
                    values="declaration_probability",
                    aggfunc="mean",
                )
                image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower")
                figure.colorbar(image, ax=axis, label="declaration probability")
                axis.set_xlabel("true FPR")
                axis.set_ylabel("sample count")
            else:
                axis.plot(frame["declaration_probability"], marker="o")
                axis.set_ylabel("declaration probability")
        else:
            y_column = next(
                (
                    name
                    for name in ("empirical_probability", "coverage", "declaration_probability")
                    if name in frame.columns
                ),
                None,
            )
            if y_column is None:
                raise ValueError("Coverage figure requires a probability column")
            x_column = next(
                (name for name in ("sample_count", "value", "true_fpr") if name in frame.columns),
                None,
            )
            if x_column is None:
                axis.plot(frame[y_column], marker="o")
            else:
                axis.plot(frame[x_column], frame[y_column], marker="o")
                axis.set_xlabel(x_column)
            axis.set_ylabel(y_column)
        axis.set_title("Experiment result")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(figure)
        if output.stat().st_size == 0:
            raise RuntimeError(f"Figure writer produced an empty file: {output}")
        return output

    @staticmethod
    def _operating_points_from_runs(
        output: Path, experiment_id: ExperimentId, outputs: OutputsLayout
    ) -> Path:
        rows = ExperimentPublisher._collect_operating_points(experiment_id, outputs)
        if not rows:
            raise ValueError("Operating-points figure requires per-client metric records")
        return build_per_client_operating_points_figure(
            output, pd.DataFrame.from_records([row.model_dump(mode="json") for row in rows])
        )

    @staticmethod
    def _collect_operating_points(
        experiment_id: ExperimentId, outputs: OutputsLayout
    ) -> list[OperatingPointRow]:
        if not outputs.runs.is_dir():
            return []
        rows: list[OperatingPointRow] = []
        for run_dir in outputs.runs.iterdir():
            rows.extend(ExperimentPublisher._run_operating_points(run_dir, experiment_id))
        return rows

    @staticmethod
    def _run_operating_points(
        run_dir: Path, experiment_id: ExperimentId
    ) -> list[OperatingPointRow]:
        if not run_dir.is_dir():
            return []
        layout = RunLayout(run_dir)
        if not layout.manifest.is_file() or not layout.metric_records.is_file():
            return []
        try:
            manifest = RunManifest.model_validate_json(layout.manifest.read_text(encoding="utf-8"))
        except pydantic.ValidationError:
            return []
        if manifest.experiment_id is not experiment_id:
            return []
        rows: list[OperatingPointRow] = []
        for line in layout.metric_records.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                record = MetricRecord.model_validate_json(line)
            except pydantic.ValidationError:
                continue
            rows.append(
                OperatingPointRow(
                    client_id=record.client_id,
                    fpr=record.fpr,
                    policy_id=record.policy_id,
                )
            )
        return rows

    @staticmethod
    def _replication_figure(
        output: Path,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
        frame: pd.DataFrame,
    ) -> Path:
        if {"client_id", "fpr"}.issubset(frame.columns):
            return build_external_replication_figure(output, frame)
        return ExperimentPublisher._operating_points_from_runs(output, experiment_id, outputs)


class ResultsManifest(BaseModel):
    model_config = Frozen

    experiment_id: ExperimentId | None = None
    complete: bool
    completion_state: CompletionState
    config_hash: Sha256
    dataset_id: DatasetId
    detector_id: DetectorId | None
    model_seeds: tuple[ModelSeed, ...]
    calibration_seeds: tuple[CalibrationSeed, ...]
    outputs_root: PathString
    file_count: PositiveCount
    source_policy: Description


@dataclass(frozen=True, slots=True)
class ResultsVerification:
    valid: bool
    problems: tuple[Description, ...]


@dataclass(frozen=True, slots=True)
class ResultsBuildResult:
    path: Path
    status: ResultsGenerationStatus


class ResultsBuilder:
    def build(
        self,
        *,
        outputs_root: Path,
        results_root: Path,
        experiment_id: ExperimentId,
        overwrite: bool = False,
        study: Study | None = None,
    ) -> Path:
        return self.build_with_status(
            outputs_root=outputs_root,
            results_root=results_root,
            experiment_id=experiment_id,
            overwrite=overwrite,
            study=study,
        ).path

    def build_with_status(
        self,
        *,
        outputs_root: Path,
        results_root: Path,
        experiment_id: ExperimentId,
        overwrite: bool = False,
        study: Study | None = None,
    ) -> ResultsBuildResult:
        resolved = study or Study.load()
        return self._build_experiment(
            experiment_id, outputs_root, results_root, overwrite=overwrite, study=resolved
        )

    def _build_experiment(
        self,
        experiment_id: ExperimentId,
        outputs_root: Path,
        results_root: Path,
        *,
        overwrite: bool,
        study: Study,
    ) -> ResultsBuildResult:
        destination = experiment_results_root(results_root, experiment_id)
        layout = ExperimentResultsBundleLayout(destination)
        if layout.manifest.is_file() and not overwrite:
            verification = ResultsVerifier().verify(results_root, experiment_id=experiment_id)
            if verification.valid:
                return ResultsBuildResult(destination, ResultsGenerationStatus.ALREADY_GENERATED)
        existed = layout.manifest.is_file()
        if destination.exists() and (overwrite or existed):
            import shutil

            shutil.rmtree(destination)
        for directory in layout.required_directories:
            directory.mkdir(parents=True, exist_ok=True)

        config = study.resolve(experiment_id)
        layout.resolved_configs.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            layout.resolved_configs / f"{experiment_id}.json",
            {
                "config_hash": config.config_hash,
                "data_spec_hash": config.data_spec_hash,
                "payload": config.model_dump(mode="json"),
            },
        )
        self._copy_experiment_delivery(experiment_id, outputs_root, layout)
        self._copy_metrics(outputs_root, layout, experiment_id=experiment_id)
        self._copy_run_artifacts(outputs_root, layout, experiment_id=experiment_id)
        self._write_experiment_provenance(experiment_id, outputs_root, config, layout)
        from fedcrg.evidence.completion import ExperimentEvidenceAssessor
        from fedcrg.evidence.contracts import experiment_contract
        from fedcrg.evidence.completion import experiment_source_files

        contract = experiment_contract(experiment_id)
        published = experiment_source_files(contract, outputs_root, experiment_id)
        source_ok = bool(published) and all(
            path.is_file() and path.stat().st_size > 0 for path in published
        )
        assessment = ExperimentEvidenceAssessor(study).assess(
            experiment_id, outputs_root=outputs_root, results_root=results_root
        )
        blocking = {
            CompletionState.NOT_STARTED,
            CompletionState.RUNNING,
            CompletionState.FAILED,
            CompletionState.INVALID,
            CompletionState.EXECUTION_INCOMPLETE,
        }
        complete = source_ok and assessment.state not in blocking
        self._write_manifest_and_checksums(
            layout,
            config,
            outputs_root,
            experiment_id=experiment_id,
            complete=complete,
            completion_state=CompletionState.ANALYSIS_COMPLETE if complete else assessment.state,
        )
        return ResultsBuildResult(
            destination,
            ResultsGenerationStatus.REBUILT
            if existed or overwrite
            else ResultsGenerationStatus.BUILT,
        )

    @staticmethod
    def _write_manifest_and_checksums(
        layout: ExperimentResultsBundleLayout,
        config: ExperimentConfig,
        outputs_root: Path,
        *,
        experiment_id: ExperimentId,
        complete: bool,
        completion_state: CompletionState,
    ) -> None:
        checksums = ResultsBuilder._checksums(layout)
        manifest = ResultsBuilder._manifest(
            config,
            outputs_root,
            checksums,
            complete,
            experiment_id=experiment_id,
            file_count=len(checksums) + 1,
            completion_state=completion_state,
        )
        atomic_write_json(layout.manifest, manifest)
        checksums = ResultsBuilder._checksums(layout)
        atomic_write_json(layout.checksums, checksums)

    @staticmethod
    def _copy_experiment_delivery(
        experiment_id: ExperimentId,
        outputs_root: Path,
        layout: ExperimentResultsBundleLayout,
    ) -> None:
        from fedcrg.evidence.contracts import experiment_contract

        contract = experiment_contract(experiment_id)
        outputs = OutputsLayout(outputs_root)
        copies = (
            (
                outputs.experiment_json(experiment_id) / LayoutArtifact.EXPERIMENT_RESULT_JSON,
                layout.json_dir / LayoutArtifact.EXPERIMENT_RESULT_JSON,
            ),
            *(
                (
                    outputs.experiment_tables(experiment_id) / os.path.basename(item.filename),
                    layout.csv_dir / os.path.basename(item.filename),
                )
                for item in contract.csv_files
            ),
            *(
                (
                    outputs.experiment_figures(experiment_id) / os.path.basename(item.filename),
                    layout.figures / os.path.basename(item.filename),
                )
                for item in contract.figure_files
            ),
        )
        for source, destination in copies:
            if not source.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

    @staticmethod
    def _write_experiment_provenance(
        experiment_id: ExperimentId,
        outputs_root: Path,
        config: ExperimentConfig,
        layout: ExperimentResultsBundleLayout,
    ) -> None:
        from fedcrg.evidence.completion import source_digests
        from fedcrg.evidence.contracts import experiment_contract

        contract = experiment_contract(experiment_id)
        provenance = ExperimentProvenance(
            experiment_id=experiment_id,
            config_hash=config.config_hash,
            data_spec_hash=config.data_spec_hash,
            source_digests=source_digests(contract, outputs_root, experiment_id),
        )
        atomic_write_json(layout.provenance_json, provenance)

    @staticmethod
    def _copy_metrics(
        outputs_root: Path,
        layout: ExperimentResultsBundleLayout,
        experiment_id: ExperimentId,
    ) -> None:
        runs_root = OutputsLayout(outputs_root).runs
        if not runs_root.exists():
            return
        records: list[MetricRecord] = []
        for run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            manifest = _run_manifest(run_root)
            if manifest is None or manifest.status is not ExperimentStatus.COMPLETE:
                continue
            if manifest.experiment_id is not experiment_id:
                continue
            run_layout = RunLayout(run_root)
            metric_path = run_layout.metric_records
            if not metric_path.is_file():
                continue
            for line in metric_path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(MetricRecord.model_validate_json(line))
                except pydantic.ValidationError:
                    continue
        atomic_write_json(
            layout.metric_records,
            {"records": [record.model_dump(mode="json") for record in records]},
        )

    @staticmethod
    def _copy_run_artifacts(
        outputs_root: Path, layout: ExperimentResultsBundleLayout, experiment_id: ExperimentId
    ) -> None:
        runs_root = OutputsLayout(outputs_root).runs
        if not runs_root.exists():
            return
        for run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            manifest = _run_manifest(run_root)
            if manifest is None or manifest.status is not ExperimentStatus.COMPLETE:
                continue
            if manifest.experiment_id is not experiment_id:
                continue
            source = RunLayout(run_root)
            target = layout.runs / run_root.name
            for artifact in (
                source.manifest,
                source.verification / LayoutArtifact.HASHES,
            ):
                if not artifact.is_file():
                    continue
                destination = target / artifact.relative_to(run_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(artifact.read_bytes())

    @staticmethod
    def _checksums(
        layout: ExperimentResultsBundleLayout,
    ) -> tuple[ChecksumRecord, ...]:
        root = layout.root
        records: list[ChecksumRecord] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == layout.checksums.name:
                continue
            relative = path.relative_to(root).as_posix()
            records.append(ChecksumRecord(relative_path=relative, sha256=sha256_file(path)))
        return tuple(records)

    @staticmethod
    def _manifest(
        config: ExperimentConfig,
        outputs_root: Path,
        checksums: tuple[ChecksumRecord, ...],
        complete: bool,
        *,
        experiment_id: ExperimentId,
        file_count: PositiveCount | None = None,
        completion_state: CompletionState,
    ) -> ResultsManifest:
        detector = config.detector
        return ResultsManifest(
            experiment_id=experiment_id,
            complete=complete,
            completion_state=completion_state,
            config_hash=config.config_hash,
            dataset_id=config.dataset.id,
            detector_id=detector.id if detector is not None else None,
            model_seeds=tuple(config.randomness.model_seeds),
            calibration_seeds=tuple(config.dataset.calibration_seeds),
            outputs_root=outputs_root.as_posix(),
            file_count=file_count if file_count is not None else len(checksums),
            source_policy="all numerical content is derived from immutable FedCRG artifacts",
        )


class ResultsVerifier:
    def verify(
        self,
        results_root: Path,
        experiment_id: ExperimentId,
    ) -> ResultsVerification:
        destination = experiment_results_root(results_root, experiment_id)
        if not destination.exists():
            return ResultsVerification(
                False, (f"experiment results bundle does not exist: {destination}",)
            )
        problems = self._verify_bundle(ExperimentResultsBundleLayout(destination))
        return ResultsVerification(not problems, tuple(problems))

    @staticmethod
    def _verify_bundle(layout: ExperimentResultsBundleLayout) -> list[Identifier]:
        destination = layout.root
        problems: list[Identifier] = []
        for directory in layout.required_directories:
            if not directory.is_dir():
                problems.append(f"missing required bundle directory: {directory.name}")
        if not layout.manifest.is_file():
            problems.append("missing bundle manifest.json")
            if not layout.checksums.is_file():
                problems.append("missing bundle checksums.json")
            return problems
        try:
            manifest = ResultsManifest.model_validate_json(
                layout.manifest.read_text(encoding="utf-8")
            )
        except pydantic.ValidationError:
            problems.append("bundle manifest.json failed schema validation")
            return problems
        if layout.manifest.stat().st_size == 0:
            problems.append("bundle manifest.json is empty")
        if not layout.checksums.is_file():
            problems.append("missing bundle checksums.json")
        else:
            problems.extend(ResultsVerifier._checksum_mismatches(destination, layout.checksums))
        problems.extend(ResultsVerifier._verify_experiment_contents(layout, manifest))
        return problems

    @staticmethod
    def _verify_experiment_contents(
        layout: ExperimentResultsBundleLayout, manifest: ResultsManifest
    ) -> list[Identifier]:
        from fedcrg.evidence.contracts import experiment_contract

        if manifest.experiment_id is None:
            return ["experiment bundle manifest is missing experiment_id"]
        contract = experiment_contract(manifest.experiment_id)
        problems: list[Identifier] = []
        if manifest.complete and manifest.completion_state not in {
            CompletionState.FULLY_PASSED,
            CompletionState.ANALYSIS_COMPLETE,
        }:
            problems.append("bundle claims complete without a passed or analysis-complete state")
        required = (
            *((layout.json_dir / item.filename, "json") for item in contract.json_files),
            *((layout.csv_dir / item.filename, "csv") for item in contract.csv_files),
            *((layout.figures / item.filename, "figure") for item in contract.figure_files),
        )
        for path, kind in required:
            problems.extend(
                ResultsVerifier._required_artifact_problems(path, kind, contract.json_kind)
            )
        problems.extend(ResultsVerifier._provenance_problems(layout))
        return problems

    @staticmethod
    def _required_artifact_problems(
        path: Path, kind: Identifier, json_kind: JsonResultKind
    ) -> list[Identifier]:
        if not path.is_file():
            return [f"missing required {kind} artifact: {path.name}"]
        if path.stat().st_size == 0:
            return [f"empty required {kind} artifact: {path.name}"]
        if kind == "json":
            return ResultsVerifier._validate_result_json(path, json_kind)
        if kind == "csv":
            return ResultsVerifier._csv_artifact_problems(path)
        return []

    @staticmethod
    def _csv_artifact_problems(path: Path) -> list[Identifier]:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeError):
            return [f"unreadable CSV artifact: {path.name}"]
        if frame.empty:
            return [f"CSV artifact has no data rows: {path.name}"]
        return []

    @staticmethod
    def _provenance_problems(layout: ExperimentResultsBundleLayout) -> list[Identifier]:
        if not layout.provenance_json.is_file():
            return ["missing experiment bundle provenance"]
        try:
            ExperimentProvenance.model_validate_json(
                layout.provenance_json.read_text(encoding="utf-8")
            )
        except pydantic.ValidationError:
            return ["experiment bundle provenance failed schema validation"]
        return []

    @staticmethod
    def _validate_result_json(path: Path, kind: JsonResultKind) -> list[Identifier]:
        try:
            payload = path.read_text(encoding="utf-8")
            parsed = json.loads(payload)
        except (OSError, json.JSONDecodeError):
            return [f"malformed JSON artifact: {path.name}"]
        if not isinstance(parsed, dict) or not parsed:
            return [f"JSON artifact has no content: {path.name}"]
        try:
            if kind is JsonResultKind.SYNTHETIC_ENVELOPE:
                envelope = SyntheticExperimentEnvelope.model_validate_json(payload)
                if not envelope.cells:
                    return [f"JSON artifact contains no cells: {path.name}"]
            elif kind is JsonResultKind.BENCHMARK_REPORT:
                report = BenchmarkReport.model_validate_json(payload)
                if not report.cells:
                    return [f"JSON artifact contains no cells: {path.name}"]
            else:
                result = EmpiricalExperimentResult.model_validate_json(payload)
                if not result.records:
                    return [f"JSON artifact contains no records: {path.name}"]
        except pydantic.ValidationError:
            return [f"JSON artifact failed typed schema: {path.name}"]
        return []

    @staticmethod
    def _checksum_mismatches(destination: Path, checksums_path: Path) -> list[Identifier]:
        listed = ResultsVerifier._load_checksums(checksums_path)
        if listed is None:
            return ["bundle checksums.json failed schema validation"]
        problems: list[Identifier] = []
        seen: set[PathString] = set()
        for path in sorted(destination.rglob("*")):
            if not path.is_file() or path.name == checksums_path.name:
                continue
            relative = path.relative_to(destination).as_posix()
            seen.add(relative)
            expected = listed.get(relative)
            if expected is None:
                problems.append(f"unchecksummed bundle file: {relative}")
            elif str(expected) != sha256_file(path):
                problems.append(f"bundle hash mismatch: {relative}")
        for relative in listed:
            if relative not in seen:
                problems.append(f"missing checksummed bundle file: {relative}")
        return problems

    @staticmethod
    def _load_checksums(checksums_path: Path) -> dict[PathString, Sha256] | None:
        try:
            return {
                record.relative_path: record.sha256
                for record in (
                    ChecksumRecord.model_validate(entry)
                    for entry in json.loads(checksums_path.read_text(encoding="utf-8"))
                )
            }
        except (OSError, json.JSONDecodeError, pydantic.ValidationError):
            return None


def build_experiment_results_bundle(
    experiment_id: ExperimentId,
    outputs_root: Path,
    results_root: Path,
    *,
    overwrite: bool = False,
) -> Path:
    return ResultsBuilder().build(
        outputs_root=outputs_root,
        results_root=results_root,
        experiment_id=experiment_id,
        overwrite=overwrite,
    )


def verify_results_bundle(
    results_root: Path,
    experiment_id: ExperimentId,
) -> ResultsVerification:
    return ResultsVerifier().verify(results_root, experiment_id=experiment_id)
