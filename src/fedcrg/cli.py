from __future__ import annotations

import platform
import shutil

import click
import numpy
import pandas
import scipy
import torch
from pydantic import BaseModel, ConfigDict

from fedcrg.config import Study, validate_experiment_config
from fedcrg.data.preparation import PrepareData
from fedcrg.hashing import sha256_file
from fedcrg.evidence.completion import ExperimentEvidenceAssessor
from fedcrg.experiments.execution import ExperimentExecutor
from fedcrg.paths import (
    OutputsLayout,
    PreparedDatasetLayout,
    prepared_dataset_family_root,
)
from fedcrg.reporting import (
    ResultsBuilder,
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
    CompletionState,
    DatasetId,
    Description,
    ExecutionOutcome,
    ExperimentId,
    ExperimentType,
    Identifier,
    ModelSeed,
    NonNegativeCount,
    PathString,
    DeviceName,
    PolicyId,
    ResultsGenerationStatus,
    Sha256,
    Version,
)

_DATASET_EXPERIMENTS: dict[DatasetId, ExperimentId] = {
    DatasetId.NBAIOT: ExperimentId.PRIMARY_NBAIOT,
    DatasetId.DIAD: ExperimentId.EXTERNAL_DIAD,
}


class DoctorPayload(BaseModel):
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
    model_config = ConfigDict(frozen=True)

    valid: bool
    experiment: ExperimentId
    category: ExperimentType
    config_hash: Sha256


class PlanPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment: ExperimentId
    category: ExperimentType
    config_hash: Sha256
    model_seeds: tuple[ModelSeed, ...]
    calibration_seeds: tuple[CalibrationSeed, ...]
    policies: tuple[PolicyId, ...]
    dependencies: tuple[ExperimentId, ...]


class PreprocessPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: DatasetId
    cache_root: PathString
    data_spec_hash: Sha256
    manifest_sha256: Sha256


class RunPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment: ExperimentId
    status: ExecutionOutcome
    completion: CompletionState
    json_results: tuple[PathString, ...]
    csv_results: tuple[PathString, ...]
    figures: tuple[PathString, ...]
    result_bundle: PathString
    model_count: NonNegativeCount
    run_directory_count: NonNegativeCount
    output: PathString


class ExperimentStatusRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment: ExperimentId
    state: CompletionState
    problems: tuple[Description, ...]
    complete_run_count: NonNegativeCount
    expected_run_count: NonNegativeCount


class StatusPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiments: tuple[ExperimentStatusRow, ...]


class ResultsBuildPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: ResultsGenerationStatus
    results_path: PathString


class ResultsVerifyPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid: bool
    problems: tuple[Description, ...]


def _study(ctx: click.Context) -> Study:
    study = ctx.obj
    assert isinstance(study, Study)
    return study


def _print(payload: BaseModel) -> None:
    click.echo(payload.model_dump_json(indent=2))


@click.group()
@click.version_option(package_name="fedcrg")
@click.pass_context
def cli(ctx: click.Context) -> None:
    study = Study.load()
    ctx.obj = study
    configure_logging(logs_root=OutputsLayout(study.paths.outputs_root).logs)


@cli.command(name="doctor")
def doctor() -> None:
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
    study = _study(ctx)
    experiment = ExperimentId(experiment_id)
    result = ExperimentExecutor(study=study).execute(experiment, overwrite=overwrite)
    _print(
        RunPayload(
            experiment=result.experiment_id,
            status=result.outcome,
            completion=result.state,
            json_results=result.json_paths,
            csv_results=result.csv_paths,
            figures=result.figure_paths,
            result_bundle=result.bundle_path,
            model_count=result.model_count,
            run_directory_count=result.run_directory_count,
            output=result.output_root,
        )
    )


@cli.command(name="status")
@click.argument(
    "experiment_id",
    required=False,
    type=click.Choice([member.value for member in ExperimentId]),
)
@click.pass_context
def status(ctx: click.Context, experiment_id: str | None) -> None:
    study = _study(ctx)
    assessor = ExperimentEvidenceAssessor(study)
    selected = (
        (ExperimentId(experiment_id),)
        if experiment_id is not None
        else tuple(spec.id for spec in study.catalogue.all())
    )
    rows = tuple(
        ExperimentStatusRow(
            experiment=item,
            state=assessment.state,
            problems=tuple(problem.detail for problem in assessment.problems),
            complete_run_count=assessment.complete_run_count,
            expected_run_count=assessment.expected_run_count,
        )
        for item in selected
        for assessment in (assessor.assess(item),)
    )
    _print(StatusPayload(experiments=rows))


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


@cli.group(name="results", invoke_without_command=True)
@click.option(
    "--overwrite", is_flag=True, help="Rebuild the delivery bundle from verified sources."
)
@click.argument("experiment_id", type=click.Choice([member.value for member in ExperimentId]))
@click.pass_context
def results_group(ctx: click.Context, overwrite: bool, experiment_id: str) -> None:
    if ctx.invoked_subcommand is not None:
        return
    study = _study(ctx)
    built = ResultsBuilder().build_with_status(
        outputs_root=study.paths.outputs_root,
        results_root=study.paths.results_root,
        experiment_id=ExperimentId(experiment_id),
        overwrite=overwrite,
        study=study,
    )
    _print(ResultsBuildPayload(status=built.status, results_path=built.path.as_posix()))


@results_group.command(name="verify")
@click.argument("experiment_id", type=click.Choice([member.value for member in ExperimentId]))
@click.pass_context
def results_verify(ctx: click.Context, experiment_id: str) -> None:
    study = _study(ctx)
    verification = verify_results_bundle(
        study.paths.results_root,
        experiment_id=ExperimentId(experiment_id),
    )
    _print(ResultsVerifyPayload(valid=verification.valid, problems=tuple(verification.problems)))
    if not verification.valid:
        raise click.ClickException("Results verification failed")


cli.add_command(results_group)


if __name__ == "__main__":
    cli()
