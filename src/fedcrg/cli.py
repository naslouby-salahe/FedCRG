"""The `fedcrg` command-line interface."""

from __future__ import annotations

import platform
import shutil
from pathlib import Path

import click
import numpy
import pandas
import scipy
import torch
from pydantic import BaseModel, ConfigDict, ValidationError

from fedcrg.config import Study, validate_experiment_config
from fedcrg.data.preparation import PrepareData
from fedcrg.evidence.models import RunConfig, RunManifest
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
from fedcrg.paths import (
    ConfigLayout,
    OutputsLayout,
    PreparedDatasetLayout,
    RunLayout,
    prepared_dataset_family_root,
)
from fedcrg.reporting import (
    build_experiment_results_bundle,
    build_publication,
    build_repository_report,
    build_results_bundle,
    ensure_experiment_results_bundle,
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
    CampaignStage,
    DatasetId,
    Description,
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

_SYNTHETIC_CATEGORIES = frozenset({ExperimentType.SYNTHETIC, ExperimentType.BENCHMARK})

_DATASET_EXPERIMENTS: dict[DatasetId, ExperimentId] = {
    DatasetId.NBAIOT: ExperimentId.PRIMARY_NBAIOT,
    DatasetId.DIAD: ExperimentId.EXTERNAL_DIAD,
}


class DoctorPayload(BaseModel):
    """Output of `fedcrg doctor`: environment and CUDA availability."""

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
    """Output of `fedcrg validate`."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    experiment: ExperimentId
    category: ExperimentType
    config_hash: Sha256


class PlanPayload(BaseModel):
    """Output of `fedcrg plan`."""

    model_config = ConfigDict(frozen=True)

    experiment: ExperimentId
    category: ExperimentType
    config_hash: Sha256
    model_seeds: tuple[ModelSeed, ...]
    calibration_seeds: tuple[CalibrationSeed, ...]
    policies: tuple[PolicyId, ...]
    dependencies: tuple[ExperimentId, ...]


class PreprocessPayload(BaseModel):
    """Output of `fedcrg preprocess`."""

    model_config = ConfigDict(frozen=True)

    dataset: DatasetId
    cache_root: PathString
    data_spec_hash: Sha256
    manifest_sha256: Sha256


class RunPayload(BaseModel):
    """Output of `fedcrg run`."""

    model_config = ConfigDict(frozen=True)

    experiment: ExperimentId
    model_count: NonNegativeCount
    run_directory_count: NonNegativeCount
    output: PathString


class CampaignPayload(BaseModel):
    """Output of `fedcrg campaign`."""

    model_config = ConfigDict(frozen=True)

    status: Identifier
    completed: NonNegativeCount
    total: PositiveCount
    current_experiment: ExperimentId | None
    elapsed_seconds: Duration
    results_path: PathString | None


class RunStatusCounts(BaseModel):
    """Run counts by status for one experiment, used to build `StatusPayload`."""

    model_config = ConfigDict()

    total: NonNegativeCount = 0
    completed: NonNegativeCount = 0
    failed: NonNegativeCount = 0
    running: NonNegativeCount = 0


class ExperimentStatusRow(BaseModel):
    """One experiment's row in `StatusPayload`."""

    model_config = ConfigDict(frozen=True)

    experiment: ExperimentId
    total: NonNegativeCount
    completed: NonNegativeCount
    failed: NonNegativeCount
    running: NonNegativeCount


class StatusPayload(BaseModel):
    """Output of `fedcrg status`."""

    model_config = ConfigDict(frozen=True)

    campaign_stage: Identifier | None
    experiments: tuple[ExperimentStatusRow, ...]


class ReportPayload(BaseModel):
    """Output of `fedcrg report`."""

    model_config = ConfigDict(frozen=True)

    repository_report: PathString
    publication_manifest: PathString


class ResultsBuildPayload(BaseModel):
    """Output of `fedcrg results build`."""

    model_config = ConfigDict(frozen=True)

    results_path: PathString


class ResultsVerifyPayload(BaseModel):
    """Output of `fedcrg results verify`."""

    model_config = ConfigDict(frozen=True)

    valid: bool
    problems: tuple[Description, ...]


def _study(ctx: click.Context) -> Study:
    """Fetch the `Study` stashed on the Click context by the top-level group callback."""
    study = ctx.obj
    assert isinstance(study, Study)
    return study


def _print(payload: BaseModel) -> None:
    """Print a payload as indented JSON."""
    click.echo(payload.model_dump_json(indent=2))


def _precompute_protocol_tables(study: Study, experiment_id: ExperimentId) -> None:
    """Precompute and cache the readiness/mismatch lookup tables an experiment depends on."""
    spec = study.spec(experiment_id)
    config = study.resolve(experiment_id)
    ProtocolTablePrecomputer().precompute(config, spec)


def _purge_experiment_evidence(experiment_id: ExperimentId, outputs_root: Path) -> None:
    """Delete run directories belonging to an experiment, ahead of an `--overwrite` re-run."""
    runs_root = OutputsLayout(outputs_root).runs
    if not runs_root.is_dir():
        return
    for run_dir in runs_root.iterdir():
        run_config_path = RunLayout(run_dir).run_config
        if not run_config_path.is_file():
            continue
        try:
            run_config = RunConfig.model_validate_json(run_config_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            # An unreadable run_config can't be attributed to this experiment; leave it alone
            # rather than guess, so --overwrite never deletes evidence it can't identify.
            continue
        if run_config.experiment_id is experiment_id:
            shutil.rmtree(run_dir, ignore_errors=True)


@click.group()
@click.version_option(package_name="fedcrg")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Load the study configuration and configure logging before any subcommand runs."""
    study = Study.load()
    ctx.obj = study
    configure_logging(logs_root=OutputsLayout(study.paths.outputs_root).logs)


@cli.command(name="doctor")
def doctor() -> None:
    """Report library versions and CUDA availability."""
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
@click.argument("experiment_id", type=click.Choice([member.value for member in ExperimentId]))
@click.pass_context
def validate(ctx: click.Context, experiment_id: str) -> None:
    """Resolve and validate an experiment's configuration without running it."""
    study = _study(ctx)
    experiment = ExperimentId(experiment_id)
    spec = study.spec(experiment)
    config = study.resolve(experiment)
    try:
        validate_experiment_config(config)
    except Exception as exc:
        raise click.ClickException(f"Experiment {experiment} is invalid: {exc}") from exc
    _print(
        ValidationPayload(
            valid=True,
            experiment=experiment,
            category=spec.category,
            config_hash=config.config_hash,
        )
    )


@cli.command(name="plan")
@click.argument("experiment_id", type=click.Choice([member.value for member in ExperimentId]))
@click.pass_context
def plan(ctx: click.Context, experiment_id: str) -> None:
    """Show what an experiment would run: seeds, policies, and dependencies."""
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
    type=click.Choice([member.value for member in DatasetId]),
)
@click.option("--overwrite", is_flag=True, help="Rebuild the prepared cache explicitly.")
@click.pass_context
def preprocess(ctx: click.Context, dataset_id: str | None, overwrite: bool) -> None:
    """Prepare (or reuse the cached preparation of) one or all raw datasets."""
    study = _study(ctx)
    if dataset_id is None:
        datasets = tuple(_DATASET_EXPERIMENTS)
    else:
        dataset = DatasetId(dataset_id)
        if dataset not in _DATASET_EXPERIMENTS:
            raise click.BadParameter(
                f"Raw dataset {dataset_id!r} has no preprocessing pipeline, "
                "expected one of " + ", ".join(sorted(_DATASET_EXPERIMENTS))
            )
        datasets = (dataset,)
    preparer = PrepareData()
    for dataset in datasets:
        experiment_id = _DATASET_EXPERIMENTS[dataset]
        config = study.resolve(experiment_id)
        if overwrite:
            dataset_root = prepared_dataset_family_root(config.preprocessed_root, dataset)
            if dataset_root.is_dir():
                shutil.rmtree(dataset_root)
        manifest = preparer.ensure_prepared(config, study.paths.data_root)
        cache_root = preparer.cache_root(config, manifest)
        _print(
            PreprocessPayload(
                dataset=dataset,
                cache_root=cache_root.as_posix(),
                data_spec_hash=config.data_spec_hash,
                manifest_sha256=sha256_file(PreparedDatasetLayout(cache_root).manifest),
            )
        )


@cli.command(name="run")
@click.argument("experiment_id", type=click.Choice([member.value for member in ExperimentId]))
@click.option("--overwrite", is_flag=True, help="Re-run and replace regenerable evidence.")
@click.pass_context
def run(ctx: click.Context, experiment_id: str, overwrite: bool) -> None:
    """Run one experiment end to end, writing run summaries and refreshing the repository report, reusing cached artifacts unless `--overwrite` is set."""
    study = _study(ctx)
    experiment = ExperimentId(experiment_id)
    spec = study.spec(experiment)
    config = study.resolve(experiment)
    outputs_root = config.outputs_root
    if overwrite:
        _purge_experiment_evidence(experiment, outputs_root)
    layout = OutputsLayout(outputs_root)
    if spec.category in _SYNTHETIC_CATEGORIES:
        _precompute_protocol_tables(study, experiment)
        output = layout.analysis_result(experiment)
        if overwrite and output.is_file():
            output.unlink()
        if experiment is ExperimentId.COMPUTATIONAL_BENCHMARK:
            RunBenchmark(spec, config).run(output)
        else:
            RunSyntheticExperiments().run(experiment, spec, config, output)
        ensure_experiment_results_bundle(experiment, outputs_root, study.paths.results_root)
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
    build_repository_report(outputs_root, config)
    ensure_experiment_results_bundle(experiment, outputs_root, study.paths.results_root)
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
    """Run every experiment in the catalogue in dependency order."""
    study = _study(ctx)
    outputs_root = study.paths.outputs_root
    store = CampaignStatusStore(outputs_root=outputs_root)
    if overwrite:
        status_path = store.path_for()
        if status_path.is_file():
            status_path.unlink()
    work_items = tuple(
        CampaignWorkItem(
            experiment_id=spec.id,
            config_path=ConfigLayout().study,
            prepared_root=study.paths.preprocessed_root,
        )
        for spec in study.catalogue.all()
    )
    status = CampaignExecutor(study=study).run(
        work_items,
        outputs_root=outputs_root,
        results_root=study.paths.results_root,
    )
    _print(
        CampaignPayload(
            status=status.current_stage if status.current_stage else CampaignStage.PENDING,
            completed=len(status.completed_experiments),
            total=max(1, len(work_items)),
            current_experiment=status.current_experiment,
            elapsed_seconds=status.elapsed_seconds,
            results_path=status.results_path,
        )
    )


def _collect_run_status_counts(runs_root: Path) -> dict[ExperimentId, RunStatusCounts]:
    """Tally run status per experiment by reading each run directory's manifest."""
    counts: dict[ExperimentId, RunStatusCounts] = {}
    if not runs_root.is_dir():
        return counts
    for run_dir in runs_root.iterdir():
        run_layout = RunLayout(run_dir)
        if not run_layout.manifest.is_file():
            continue
        try:
            manifest = RunManifest.model_validate_json(
                run_layout.manifest.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError):
            continue
        counter = counts.setdefault(manifest.experiment_id, RunStatusCounts())
        counter.total += 1
        if manifest.status is ExperimentStatus.COMPLETE:
            counter.completed += 1
        elif manifest.status is ExperimentStatus.FAILED:
            counter.failed += 1
        else:
            counter.running += 1
    return counts


@cli.command(name="status")
@click.argument(
    "experiment_id",
    required=False,
    type=click.Choice([member.value for member in ExperimentId]),
)
@click.pass_context
def status(ctx: click.Context, experiment_id: str | None) -> None:
    """Show run counts by status, for one experiment or all of them."""
    study = _study(ctx)
    outputs_root = study.paths.outputs_root
    layout = OutputsLayout(outputs_root)
    counts = _collect_run_status_counts(layout.runs)
    rows = tuple(
        ExperimentStatusRow(
            experiment=experiment,
            total=values.total,
            completed=values.completed,
            failed=values.failed,
            running=values.running,
        )
        for experiment, values in sorted(counts.items(), key=lambda item: item[0])
        if experiment_id is None or experiment == experiment_id
    )
    campaign_stage: Identifier | None = None
    try:
        campaign_status = CampaignStatusStore(outputs_root=outputs_root).load()
        campaign_stage = campaign_status.current_stage if campaign_status.current_stage else None
    except FileNotFoundError:
        pass
    _print(
        StatusPayload(
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
    """Stream resource telemetry to stdout and a log file until interrupted."""
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
    """Build the repository hygiene report and publication manifest."""
    study = _study(ctx)
    config = study.resolve(ExperimentId.PRIMARY_NBAIOT)
    outputs_root = study.paths.outputs_root
    repository_report = build_repository_report(outputs_root, config)
    publication_manifest = build_publication(config, outputs_root)
    _print(
        ReportPayload(
            repository_report=repository_report.as_posix(),
            publication_manifest=publication_manifest.as_posix(),
        )
    )


@cli.group(name="results")
def results_group() -> None:
    """Commands for building and verifying packaged results bundles."""


@results_group.command(name="build")
@click.argument(
    "experiment_id",
    required=False,
    type=click.Choice([member.value for member in ExperimentId]),
)
@click.pass_context
def results_build(ctx: click.Context, experiment_id: str | None) -> None:
    """Build the campaign results bundle, or one experiment's bundle when an experiment id is given."""
    study = _study(ctx)
    if experiment_id is None:
        path = build_results_bundle(
            outputs_root=study.paths.outputs_root,
            results_root=study.paths.results_root,
        )
    else:
        path = build_experiment_results_bundle(
            ExperimentId(experiment_id),
            outputs_root=study.paths.outputs_root,
            results_root=study.paths.results_root,
        )
    _print(ResultsBuildPayload(results_path=path.as_posix()))


@results_group.command(name="verify")
@click.argument(
    "experiment_id",
    required=False,
    type=click.Choice([member.value for member in ExperimentId]),
)
@click.pass_context
def results_verify(ctx: click.Context, experiment_id: str | None) -> None:
    """Verify the campaign bundle and every experiment bundle, or one experiment bundle."""
    study = _study(ctx)
    verification = verify_results_bundle(
        study.paths.results_root,
        experiment_id=ExperimentId(experiment_id) if experiment_id is not None else None,
    )
    _print(ResultsVerifyPayload(valid=verification.valid, problems=tuple(verification.problems)))
    if not verification.valid:
        raise click.ClickException("Results verification failed")


cli.add_command(results_group)


if __name__ == "__main__":
    cli()
