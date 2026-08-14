"""FedCRG research command-line interface.

One click entry point covering configuration validation, dataset
preprocessing, experiment planning and execution, campaign orchestration,
status inspection, resource monitoring, reports, and publication results
bundles. CLI options are the input boundary: values are converted to typed
identities before any scientific call, and every printed payload is a
validated Pydantic model serialized with ``model_dump_json``.
"""

from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

import click
import numpy
import pandas
import scipy
import torch
from pydantic import BaseModel, ConfigDict

from fedcrg.config import Study, validate_experiment_config
from fedcrg.data.preparation import PrepareData
from fedcrg.evidence.store import OutputsLayout, PreparedLayout
from fedcrg.hashing import sha256_file
from fedcrg.experiments.analyses import (
    ProtocolTablePrecomputer,
    RunBenchmark,
    RunSyntheticExperiments,
)
from fedcrg.experiments.runner import (
    CampaignExecutor,
    CampaignStatusStore,
    CampaignWorkItem,
    RunAllExperiments,
)
from fedcrg.reporting import (
    build_publication,
    build_repository_report,
    build_results_bundle,
    verify_results_bundle,
)
from fedcrg.runtime import (
    ResourceMonitor,
    configure_logging,
    render_telemetry,
    write_telemetry,
)
from fedcrg.types import (
    CalibrationSeed,
    CampaignId,
    CampaignStage,
    DatasetId,
    Duration,
    ExperimentId,
    ExperimentStatus,
    ExperimentType,
    Identifier,
    ModelSeed,
    NonNegativeCount,
    PathString,
    DeviceName,
    PolicyId,
    PositiveCount,
    Sha256,
    Version,
)

from fedcrg.evidence.store import RunLayout

_SYNTHETIC_CATEGORIES = frozenset({ExperimentType.SYNTHETIC, ExperimentType.BENCHMARK})

_DATASET_EXPERIMENTS: dict[DatasetId, ExperimentId] = {
    DatasetId.NBAIOT: ExperimentId.PRIMARY_NBAIOT,
    DatasetId.DIAD: ExperimentId.EXTERNAL_DIAD,
}


class DoctorPayload(BaseModel):
    """Runtime and CUDA pin printed by the doctor command."""

    model_config = ConfigDict(frozen=True)

    python: Version
    numpy: Version
    scipy: Version
    pandas: Version
    torch: Version
    cuda_available: bool
    cuda: Identifier | None
    gpu: DeviceName | None


class ValidationPayload(BaseModel):
    """Typed result of the experiment validate command."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    experiment: ExperimentId
    category: ExperimentType
    config_hash: Sha256


class PlanPayload(BaseModel):
    """Typed result of the experiment plan command."""

    model_config = ConfigDict(frozen=True)

    experiment: ExperimentId
    category: ExperimentType
    config_hash: Sha256
    model_seeds: tuple[ModelSeed, ...]
    calibration_seeds: tuple[CalibrationSeed, ...]
    policies: tuple[PolicyId, ...]
    dependencies: tuple[ExperimentId, ...]


class PreprocessPayload(BaseModel):
    """Prepared-cache location and identity pin."""

    model_config = ConfigDict(frozen=True)

    dataset: DatasetId
    cache_root: PathString
    data_spec_hash: Sha256
    manifest_sha256: Sha256


class RunPayload(BaseModel):
    """Summary of one executed experiment workload."""

    model_config = ConfigDict(frozen=True)

    experiment: ExperimentId
    model_count: NonNegativeCount
    run_directory_count: NonNegativeCount
    output: PathString


class CampaignPayload(BaseModel):
    """Typed campaign status printed by the campaign command."""

    model_config = ConfigDict(frozen=True)

    campaign_id: CampaignId
    status: Identifier
    completed: NonNegativeCount
    total: PositiveCount
    current_experiment: ExperimentId | None
    elapsed_seconds: Duration
    results_path: PathString | None


class RunStatusCounts(BaseModel):
    """Aggregated run-status counters (structural counters, not configuration)."""

    model_config = ConfigDict()

    total: NonNegativeCount = 0
    completed: NonNegativeCount = 0
    failed: NonNegativeCount = 0
    running: NonNegativeCount = 0


class ExperimentStatusRow(BaseModel):
    """Run-status summary for one experiment."""

    model_config = ConfigDict(frozen=True)

    experiment: ExperimentId
    total: NonNegativeCount
    completed: NonNegativeCount
    failed: NonNegativeCount
    running: NonNegativeCount


class StatusPayload(BaseModel):
    """Typed status printed by the status command."""

    model_config = ConfigDict(frozen=True)

    campaign_id: CampaignId | None
    campaign_stage: Identifier | None
    experiments: tuple[ExperimentStatusRow, ...]


class ReportPayload(BaseModel):
    """Typed report build summary."""

    model_config = ConfigDict(frozen=True)

    campaign_id: CampaignId
    repository_report: PathString
    publication_manifest: PathString


class ResultsBuildPayload(BaseModel):
    """Results bundle build summary."""

    model_config = ConfigDict(frozen=True)

    results_path: PathString


class ResultsVerifyPayload(BaseModel):
    """Results bundle verification summary."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    problems: tuple[Identifier, ...]


def _study(ctx: click.Context) -> Study:
    """Return the study configuration loaded by the group callback."""
    study = ctx.obj
    assert isinstance(study, Study)
    return study


def _print(payload: BaseModel) -> None:
    click.echo(payload.model_dump_json(indent=2))


def _precompute_protocol_tables(study: Study, experiment_id: ExperimentId) -> None:
    """Freeze the readiness/mismatch protocol tables required by policy evaluation."""
    spec = study.spec(experiment_id)
    config = study.resolve(experiment_id)
    ProtocolTablePrecomputer().precompute(config, spec)


def _purge_experiment_evidence(experiment_id: ExperimentId, outputs_root: Path) -> None:
    """Remove regenerable run evidence for one experiment (explicit overwrite)."""
    runs_root = OutputsLayout(outputs_root).runs
    if not runs_root.is_dir():
        return
    for run_dir in runs_root.iterdir():
        run_config = RunLayout(run_dir).run_config
        if not run_config.is_file():
            continue
        try:
            payload = json.loads(run_config.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("experiment_id") == experiment_id.value:
            shutil.rmtree(run_dir, ignore_errors=True)


@click.group()
@click.version_option(package_name="fedcrg")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """FedCRG reproducible research tooling."""
    study = Study.load()
    ctx.obj = study
    configure_logging(logs_root=OutputsLayout(study.paths.outputs_root).logs)


@cli.command(name="doctor")
def doctor() -> None:
    """Print installed library versions and CUDA availability as JSON."""
    cuda_available = torch.cuda.is_available()
    _print(
        DoctorPayload(
            python=platform.python_version(),
            numpy=numpy.__version__,
            scipy=scipy.__version__,
            pandas=pandas.__version__,
            torch=torch.__version__,
            cuda_available=cuda_available,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if cuda_available else None,
        )
    )


@cli.command(name="validate")
@click.argument("experiment_id", type=click.Choice([item.value for item in ExperimentId]))
@click.pass_context
def validate(ctx: click.Context, experiment_id: str) -> None:
    """Validate one resolved experiment configuration."""
    study = _study(ctx)
    experiment = ExperimentId(experiment_id)
    spec = study.spec(experiment)
    config = study.resolve(experiment)
    try:
        validate_experiment_config(config)
    except Exception as exc:
        raise click.ClickException(f"Experiment {experiment.value} is invalid: {exc}") from exc
    _print(
        ValidationPayload(
            valid=True,
            experiment=experiment,
            category=spec.category,
            config_hash=config.config_hash,
        )
    )


@cli.command(name="plan")
@click.argument("experiment_id", type=click.Choice([item.value for item in ExperimentId]))
@click.pass_context
def plan(ctx: click.Context, experiment_id: str) -> None:
    """Print the execution plan for one experiment."""
    study = _study(ctx)
    experiment = ExperimentId(experiment_id)
    spec = study.spec(experiment)
    config = study.resolve(experiment)
    _print(
        PlanPayload(
            experiment=experiment,
            category=spec.category,
            config_hash=config.config_hash,
            model_seeds=config.randomness.model_seeds,
            calibration_seeds=config.dataset.calibration_seeds,
            policies=config.policies,
            dependencies=spec.dependencies,
        )
    )


@cli.command(name="preprocess")
@click.argument(
    "dataset_id",
    required=False,
    type=click.Choice([item.value for item in DatasetId]),
)
@click.option("--overwrite", is_flag=True, help="Rebuild the prepared cache explicitly.")
@click.pass_context
def preprocess(ctx: click.Context, dataset_id: str | None, overwrite: bool) -> None:
    """Preprocess raw datasets into data/preprocessed/ (reuse-first).

    Without a dataset argument every raw dataset with a preprocessing
    pipeline is prepared.
    """
    study = _study(ctx)
    if dataset_id is None:
        datasets = tuple(_DATASET_EXPERIMENTS)
    else:
        dataset = DatasetId(dataset_id)
        if dataset not in _DATASET_EXPERIMENTS:
            raise click.BadParameter(
                f"Raw dataset {dataset_id!r} has no preprocessing pipeline, "
                "expected one of " + ", ".join(sorted(item.value for item in _DATASET_EXPERIMENTS))
            )
        datasets = (dataset,)
    preparer = PrepareData()
    for dataset in datasets:
        experiment_id = _DATASET_EXPERIMENTS[dataset]
        config = study.resolve(experiment_id)
        if overwrite:
            dataset_root = config.preprocessed_root / dataset.value
            if dataset_root.is_dir():
                shutil.rmtree(dataset_root)
        manifest = preparer.ensure_prepared(config, study.paths.data_root)
        cache_root = preparer.cache_root(config, manifest)
        _print(
            PreprocessPayload(
                dataset=dataset,
                cache_root=cache_root.as_posix(),
                data_spec_hash=config.data_spec_hash,
                manifest_sha256=sha256_file(cache_root / PreparedLayout.manifest_filename),
            )
        )


@cli.command(name="run")
@click.argument("experiment_id", type=click.Choice([item.value for item in ExperimentId]))
@click.option("--overwrite", is_flag=True, help="Re-run and replace regenerable evidence.")
@click.pass_context
def run(ctx: click.Context, experiment_id: str, overwrite: bool) -> None:
    """Execute one pre-registered experiment."""
    study = _study(ctx)
    experiment = ExperimentId(experiment_id)
    spec = study.spec(experiment)
    config = study.resolve(experiment)
    outputs = config.outputs_root
    if overwrite:
        _purge_experiment_evidence(experiment, outputs)
    layout = OutputsLayout(outputs)
    if spec.category in _SYNTHETIC_CATEGORIES:
        _precompute_protocol_tables(study, experiment)
        output = layout.cache_analysis / f"{experiment.value}.json"
        if overwrite and output.is_file():
            output.unlink()
        if experiment is ExperimentId.COMPUTATIONAL_BENCHMARK:
            RunBenchmark(spec, config).run(output)
        else:
            RunSyntheticExperiments().run(experiment, spec, config, output)
        _print(
            RunPayload(
                experiment=experiment,
                model_count=0,
                run_directory_count=0,
                output=output.as_posix(),
            )
        )
        return
    _precompute_protocol_tables(study, experiment)
    workload = RunAllExperiments().execute(experiment, config, config.preprocessed_root)
    _print(
        RunPayload(
            experiment=experiment,
            model_count=len(workload.models),
            run_directory_count=len(workload.run_directories),
            output=layout.runs.as_posix(),
        )
    )


@cli.command(name="campaign")
@click.option("--overwrite", is_flag=True, help="Restart the campaign from scratch.")
@click.pass_context
def campaign(ctx: click.Context, overwrite: bool) -> None:
    """Execute the full experiment campaign from prepared data."""
    study = _study(ctx)
    campaign_id = study.campaign_id
    outputs = study.paths.outputs_root
    store = CampaignStatusStore(outputs_root=outputs)
    if overwrite:
        status_path = store.path_for(campaign_id)
        if status_path.is_file():
            status_path.unlink()
    work_items = tuple(
        CampaignWorkItem(
            experiment_id=spec.id,
            config_path=Path("config/study.yaml"),
            prepared_root=study.paths.preprocessed_root,
        )
        for spec in study.catalogue.all()
    )
    status = CampaignExecutor(study=study).run(
        campaign_id,
        work_items,
        outputs_root=outputs,
        results_root=study.paths.results_root,
    )
    _print(
        CampaignPayload(
            campaign_id=campaign_id,
            status=status.current_stage.value
            if status.current_stage
            else CampaignStage.PENDING.value,
            completed=len(status.completed_experiments),
            total=max(1, len(work_items)),
            current_experiment=status.current_experiment,
            elapsed_seconds=status.elapsed_seconds,
            results_path=status.results_path,
        )
    )


@cli.command(name="status")
@click.argument(
    "experiment_id",
    required=False,
    type=click.Choice([item.value for item in ExperimentId]),
)
@click.pass_context
def status(ctx: click.Context, experiment_id: str | None) -> None:
    """Show run status for one experiment (or all experiments)."""
    study = _study(ctx)
    outputs = study.paths.outputs_root
    layout = OutputsLayout(outputs)
    counts: dict[ExperimentId, RunStatusCounts] = {}
    runs_root = layout.runs
    if runs_root.is_dir():
        for run_dir in runs_root.iterdir():
            run_layout = RunLayout(run_dir)
            if not run_layout.manifest.is_file():
                continue
            try:
                from fedcrg.evidence.models import RunManifest

                manifest = RunManifest.model_validate_json(
                    run_layout.manifest.read_text(encoding="utf-8")
                )
            except Exception:
                continue
            counter = counts.setdefault(manifest.experiment_id, RunStatusCounts())
            counter.total += 1
            if manifest.status is ExperimentStatus.COMPLETE:
                counter.completed += 1
            elif manifest.status is ExperimentStatus.FAILED:
                counter.failed += 1
            else:
                counter.running += 1
    rows = tuple(
        ExperimentStatusRow(
            experiment=experiment,
            total=values.total,
            completed=values.completed,
            failed=values.failed,
            running=values.running,
        )
        for experiment, values in sorted(counts.items(), key=lambda item: item[0].value)
        if experiment_id is None or experiment.value == experiment_id
    )
    campaign_stage: Identifier | None = None
    campaign_record: CampaignId | None = None
    try:
        campaign_status = CampaignStatusStore(outputs_root=outputs).load(study.campaign_id)
        campaign_record = campaign_status.campaign_id
        campaign_stage = (
            campaign_status.current_stage.value if campaign_status.current_stage else None
        )
    except FileNotFoundError:
        pass
    _print(
        StatusPayload(
            campaign_id=campaign_record,
            campaign_stage=campaign_stage,
            experiments=rows,
        )
    )


@cli.command(name="monitor")
@click.option("--interval", type=float, default=1.0, show_default=True)
@click.option(
    "--samples",
    type=int,
    default=None,
    help="Stop after this many samples (default: stream until interrupted).",
)
@click.pass_context
def monitor(ctx: click.Context, interval: float, samples: int | None) -> None:
    """Stream CPU/RAM/GPU telemetry and persist it under outputs/monitoring/."""
    study = _study(ctx)
    telemetry_path = OutputsLayout(study.paths.outputs_root).telemetry_file
    monitor = ResourceMonitor()
    if interval <= 0:
        raise click.BadParameter("--interval must be positive")
    click.echo(f"Streaming resource telemetry to {telemetry_path} (Ctrl-C to stop).")
    try:
        for sample in monitor.stream(interval, samples):
            write_telemetry(sample, telemetry_path)
            click.echo(render_telemetry(sample))
    except KeyboardInterrupt:
        click.echo("\nStopped.")
        raise SystemExit(0) from None


@cli.command(name="report")
@click.pass_context
def report(ctx: click.Context) -> None:
    """Build the repository report and publication package from frozen evidence."""
    study = _study(ctx)
    config = study.resolve(ExperimentId.PRIMARY_NBAIOT)
    outputs = study.paths.outputs_root
    repository_report = build_repository_report(outputs, config)
    publication_manifest = build_publication(config, outputs)
    _print(
        ReportPayload(
            campaign_id=study.campaign_id,
            repository_report=repository_report.as_posix(),
            publication_manifest=publication_manifest.as_posix(),
        )
    )


@cli.group(name="results")
def results_group() -> None:
    """Build and verify publication bundles under results/<campaign-id>/."""


@results_group.command(name="build")
@click.pass_context
def results_build(ctx: click.Context) -> None:
    """Build the publication bundle from immutable evidence."""
    study = _study(ctx)
    path = build_results_bundle(
        study.campaign_id,
        outputs_root=study.paths.outputs_root,
        results_root=study.paths.results_root,
    )
    _print(ResultsBuildPayload(results_path=path.as_posix()))


@results_group.command(name="verify")
@click.pass_context
def results_verify(ctx: click.Context) -> None:
    """Verify that a publication bundle is complete, consistent, and hash-valid."""
    study = _study(ctx)
    verification = verify_results_bundle(
        study.campaign_id,
        results_root=study.paths.results_root,
        outputs_root=study.paths.outputs_root,
    )
    _print(ResultsVerifyPayload(valid=verification.valid, problems=tuple(verification.problems)))
    if not verification.valid:
        raise click.ClickException("Results verification failed")


cli.add_command(results_group)


if __name__ == "__main__":
    cli()
