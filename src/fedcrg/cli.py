"""FedCRG research command-line interface.

One click entry point covering configuration validation, data preparation,
experiment execution, campaign orchestration, table precomputation, and
reporting. CLI options are the input boundary: values are converted to
typed identities before any scientific call, and every printed payload is
an explicitly typed dictionary.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Protocol, TypedDict, cast

import click
import numpy
import pandas
from pydantic import TypeAdapter
import scipy
import torch

from fedcrg.config import ExperimentConfig, Study, validate_experiment_config
from fedcrg.evidence.store import (
    OutputsLayout,
    atomic_write_json,
    capture_environment,
    sha256_file,
)
from fedcrg.runtime import (
    ResourceMonitor,
    configure_logging,
    render_cache_status,
    render_campaign_status,
    render_telemetry,
    write_telemetry,
)
from fedcrg.thresholding.readiness import (
    ReadinessPlan,
    ReadinessPlanBuilder,
    ReadinessPlanCache,
    familywise_readiness_assurance,
)
from fedcrg.types import (
    CalibrationSeed,
    CampaignId,
    DatasetId,
    Duration,
    ExperimentId,
    ExperimentType,
    Identifier,
    ModelSeed,
    NonNegativeCount,
    PositiveCount,
    Sha256,
    Version,
)

_DEFAULT_STUDY_CONFIG = Path("config/study.yaml")
_PREPROCESSED_ROOT = Path("data/preprocessed")
_MODEL_SEED = TypeAdapter(ModelSeed)
_CALIBRATION_SEED = TypeAdapter(CalibrationSeed)
_CAMPAIGN_ID = TypeAdapter(CampaignId)


class DoctorPayload(TypedDict):
    """Runtime and CUDA pin printed by the doctor command."""

    python: Version
    numpy: Version
    scipy: Version
    pandas: Version
    torch: Version
    cuda_available: bool
    cuda: Identifier | None
    gpu: Identifier | None


class ValidationPayload(TypedDict):
    """Typed result of the experiment validate command."""

    valid: bool
    experiment: ExperimentId
    type: ExperimentType
    dependencies: tuple[ExperimentId, ...]
    config_hash: Sha256


class PlanPayload(TypedDict):
    """Typed result of the experiment plan command."""

    experiment: ExperimentId
    config_hash: Sha256
    model_seed: ModelSeed | None
    calibration_seed: CalibrationSeed
    dependencies: tuple[ExperimentId, ...]


class RunSummaryPayload(TypedDict):
    """Typed summary of one executed model workload."""

    experiment: ExperimentId
    model_seed: ModelSeed
    calibration_seed: CalibrationSeed
    config_hash: Sha256
    model_count: PositiveCount
    policy_run_count: PositiveCount


class PreparedEntry(TypedDict):
    """One prepared-cache status row."""

    dataset: DatasetId
    identity: Identifier
    prepared: bool
    manifest_sha256: Sha256 | None


class PreparedStatusPayload(TypedDict):
    """Status of every dataset in the prepared cache."""

    prepared: tuple[PreparedEntry, ...]


class CampaignStatusPayload(TypedDict):
    """Typed campaign status printed by the campaign status command."""

    campaign_id: CampaignId
    status: Identifier
    completed: NonNegativeCount
    total: PositiveCount
    current_experiment: ExperimentId | None
    elapsed_seconds: NonNegativeCount


class FreezePayload(TypedDict):
    """Environment freeze pin printed by the freeze command."""
    path: Identifier
    sha256: Sha256
    git_commit: Identifier
    git_clean: bool


class PublicationPayload(TypedDict):
    """Publication build summary printed by the report command."""
    manifest: Identifier
    complete: bool


class ResultsBuildPayload(TypedDict):
    """Results bundle build summary."""
    results_path: Identifier


class ResultsVerifyPayload(TypedDict):
    """Results bundle verification summary."""
    valid: bool
    problems: tuple[Identifier, ...]


class ReadinessTablePayload(TypedDict):
    """Precomputed readiness-plan table payload."""
    plans: tuple[ReadinessPlan, ...]


class _ModelEvidence(Protocol):
    model_seed: ModelSeed


class _WorkloadOutcome(Protocol):
    experiment_id: ExperimentId
    models: tuple[_ModelEvidence, ...]
    run_directories: tuple[Path, ...]


class _RunOutcome(Protocol):
    workload: _WorkloadOutcome


class _ExperimentRunner(Protocol):
    def execute(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        prepared_root: Path,
        *,
        model_seed: ModelSeed,
        calibration_seed: CalibrationSeed,
    ) -> _RunOutcome: ...


class _CampaignStatus(Protocol):
    status: Identifier
    completed: NonNegativeCount
    total: PositiveCount
    current_experiment: Identifier | None
    elapsed_seconds: Duration


class _CampaignStatusStore(Protocol):
    def __init__(
        self,
        campaigns_root: Path | None = None,
        outputs_root: Path = Path("outputs"),
    ) -> None: ...
    def load(self, campaign_id: CampaignId) -> _CampaignStatus: ...


class _CampaignWorkItem(Protocol):
    def __init__(
        self, experiment_id: ExperimentId, config_path: Path, prepared_root: Path
    ) -> None: ...
    experiment_id: ExperimentId
    config_path: Path
    prepared_root: Path


class _CampaignRunner(Protocol):
    def __init__(self) -> None: ...
    def run(
        self,
        campaign_id: CampaignId,
        work_items: tuple[_CampaignWorkItem, ...],
        *,
        outputs_root: Path,
        results_root: Path,
    ) -> _CampaignStatus: ...


class _CampaignModule(Protocol):
    CampaignRunner: type[_CampaignRunner]
    CampaignWorkItem: type[_CampaignWorkItem]
    CampaignStatusStore: type[_CampaignStatusStore]


class _PublicationPackage(Protocol):
    manifest: Path
    complete: bool


class _BundleVerification(Protocol):
    valid: bool
    problems: tuple[Identifier, ...]


class _Reporting(Protocol):
    def build_run_report(self, run_dir: Path) -> Path: ...
    def build_repository_report(self, outputs: Path, config: ExperimentConfig) -> Path: ...
    def build_publication(
        self,
        config: ExperimentConfig,
        outputs_root: Path,
        prepared_manifest: Path | None,
        destination: Path | None,
    ) -> _PublicationPackage: ...
    def build_results_bundle(
        self, campaign_id: CampaignId, outputs_root: Path, results_root: Path
    ) -> Path: ...
    def verify_results_bundle(
        self, campaign_id: CampaignId, results_root: Path, outputs_root: Path
    ) -> _BundleVerification: ...


class _PrepareData(Protocol):
    def prepare(self, config: ExperimentConfig, data_root: Path) -> Path: ...


class _FeatureSensitivityPreparer(Protocol):
    def prepare(
        self,
        config: ExperimentConfig,
        data_root: Path,
        eligibility_manifest: Path,
        feature_manifest: Path,
    ) -> tuple[ExperimentConfig, Path]: ...


class _DatasetPreparationModule(Protocol):
    PrepareData: type[_PrepareData]
    PrepareDiadFeatureSensitivity: type[_FeatureSensitivityPreparer]


class _SensitivityRunner(Protocol):
    def run(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        score_root: Path,
        prepared_root: Path,
        *,
        model_seed: ModelSeed | None,
        calibration_seed: CalibrationSeed | None,
        output: Path,
    ) -> Path: ...


class _SyntheticRunner(Protocol):
    def run(self, experiment_id: ExperimentId, config: ExperimentConfig, output: Path) -> Path: ...


class _RobustnessRunner(Protocol):
    def run_deep_svdd(
        self, config: ExperimentConfig, prepared_root: Path, model_seed: ModelSeed
    ) -> tuple[Path, Path]: ...


class _BenchmarkRunner(Protocol):
    def run(self, config: ExperimentConfig, output: Path) -> Path: ...


def _load_study(config_path: Path | None) -> Study:
    if config_path is not None:
        return Study.load(study_path=config_path)
    return Study.load()


def _experiment_runner() -> _ExperimentRunner:
    try:
        from fedcrg.experiments.runner import (  # pyright: ignore[reportMissingImports]  # wired after runner lands
            RunAllExperiments,
        )
    except ImportError as exc:
        raise click.ClickException(
            "The experiment execution layer is not wired into this build yet."
        ) from exc
    return cast(_ExperimentRunner, RunAllExperiments())


def _campaign_module() -> _CampaignModule:
    try:
        import fedcrg.experiments.runner as campaign  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise click.ClickException(
            "The campaign execution layer is not wired into this build yet."
        ) from exc
    return cast(_CampaignModule, campaign)


def _reporting() -> _Reporting:
    try:
        import fedcrg.reporting as reporting  # pyright: ignore[reportMissingImports]  # wired after runner lands
    except ImportError as exc:
        raise click.ClickException(
            "The reporting layer is not wired into this build yet."
        ) from exc
    return cast(_Reporting, reporting)


def _dataset_preparation() -> _DatasetPreparationModule:
    try:
        import fedcrg.experiments.dataset_preparation as dataset_preparation  # pyright: ignore[reportMissingImports]  # wired after runner lands
    except ImportError as exc:
        raise click.ClickException(
            "The data preparation layer is not wired into this build yet."
        ) from exc
    return cast(_DatasetPreparationModule, dataset_preparation)


def _sensitivity_runner() -> _SensitivityRunner:
    try:
        from fedcrg.experiments.definitions.sensitivity import (  # pyright: ignore[reportMissingImports]  # wired after runner lands
            RunRealSensitivities,
        )
    except ImportError as exc:
        raise click.ClickException(
            "The sensitivity layer is not wired into this build yet."
        ) from exc
    return cast(_SensitivityRunner, RunRealSensitivities())


def _synthetic_runner() -> _SyntheticRunner:
    try:
        from fedcrg.experiments.definitions.synthetic import (  # pyright: ignore[reportMissingImports]  # wired after runner lands
            RunSyntheticExperiments,
        )
    except ImportError as exc:
        raise click.ClickException(
            "The synthetic layer is not wired into this build yet."
        ) from exc
    return cast(_SyntheticRunner, RunSyntheticExperiments())


def _robustness_runner() -> _RobustnessRunner:
    try:
        from fedcrg.experiments.model_training import (  # pyright: ignore[reportMissingImports]  # wired after runner lands
            TrainDetector,
        )
    except ImportError as exc:
        raise click.ClickException(
            "The detector training layer is not wired into this build yet."
        ) from exc
    return cast(_RobustnessRunner, TrainDetector())


def _benchmark_runner() -> _BenchmarkRunner:
    try:
        from fedcrg.experiments.computational_benchmark import (  # pyright: ignore[reportMissingImports]  # wired after runner lands
            RunBenchmark,
        )
    except ImportError as exc:
        raise click.ClickException(
            "The benchmark layer is not wired into this build yet."
        ) from exc
    return cast(_BenchmarkRunner, RunBenchmark())


@click.group()
@click.version_option(package_name="fedcrg")
def cli() -> None:
    """FedCRG reproducible research tooling."""
    configure_logging(logs_root=Path("outputs/logs"))


@cli.command(name="doctor")
def doctor() -> None:
    """Print installed library versions and CUDA availability as JSON."""
    cuda_available = torch.cuda.is_available()
    payload = DoctorPayload(
        python=platform.python_version(),
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        pandas=pandas.__version__,
        torch=torch.__version__,
        cuda_available=cuda_available,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if cuda_available else None,
    )
    click.echo(json.dumps(payload, indent=2))


@cli.command(name="monitor")
@click.option(
    "--outputs",
    type=click.Path(path_type=Path),
    default=Path("outputs"),
    show_default=True,
)
@click.option(
    "--interval",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds between samples.",
)
@click.option(
    "--samples",
    type=int,
    default=None,
    help="Stop after this many samples (default: stream until interrupted).",
)
def monitor_command(outputs: Path, interval: float, samples: int | None) -> None:
    """Stream CPU/RAM/GPU telemetry and persist it under outputs/monitoring/."""
    telemetry_path = OutputsLayout(outputs).telemetry_file
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


@cli.command(name="run-experiment")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option(
    "--experiment",
    type=click.Choice([item.value for item in ExperimentId]),
    required=True,
)
@click.option("--model-seed", type=int, required=True)
@click.option("--calibration-seed", type=int, required=True)
@click.option(
    "--prepared-root",
    type=click.Path(path_type=Path),
    default=str(_PREPROCESSED_ROOT),
    show_default=True,
)
def run_experiment(
    config_path: Path | None,
    experiment: str,
    model_seed: int,
    calibration_seed: int,
    prepared_root: Path,
) -> None:
    """Execute one pre-registered experiment cell from prepared data.

    Trains one model seed, scores it, and materializes every configured
    policy run for the given calibration seed.
    """
    study = _load_study(config_path)
    experiment_id = ExperimentId(experiment)
    config = study.resolve(experiment_id)
    typed_model_seed = _MODEL_SEED.validate_python(int(model_seed))
    typed_calibration_seed = _CALIBRATION_SEED.validate_python(int(calibration_seed))
    outcome = _experiment_runner().execute(
        experiment_id,
        config,
        prepared_root,
        model_seed=typed_model_seed,
        calibration_seed=typed_calibration_seed,
    )
    workload = outcome.workload
    payload = RunSummaryPayload(
        experiment=experiment_id,
        model_seed=typed_model_seed,
        calibration_seed=typed_calibration_seed,
        config_hash=config.config_hash,
        model_count=len(workload.models),
        policy_run_count=len(workload.run_directories),
    )
    click.echo(json.dumps(payload, indent=2))


@cli.command(name="benchmark")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
)
def benchmark_command(config_path: Path | None, output: Path | None) -> None:
    """Run the computational benchmark on synthetic evidence."""
    study = _load_study(config_path)
    config = study.resolve(ExperimentId.COMPUTATIONAL_BENCHMARK)
    target = output or OutputsLayout(config.outputs_root).benchmark_report
    path = _benchmark_runner().run(config, target)
    click.echo(str(path))


@click.group(name="experiment")
def experiment_group() -> None:
    """Validate and plan typed experiment executions."""


@experiment_group.command(name="validate")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option(
    "--experiment",
    type=click.Choice([item.value for item in ExperimentId]),
    required=True,
)
def validate_experiment(config_path: Path | None, experiment: str) -> None:
    """Validate a resolved experiment configuration against the catalogue."""
    study = _load_study(config_path)
    experiment_id = ExperimentId(experiment)
    spec = study.spec(experiment_id)
    config = study.resolve(experiment_id)
    try:
        validate_experiment_config(config)
    except Exception as exc:
        raise click.ClickException(f"Experiment {experiment_id.value} is invalid: {exc}") from exc
    payload = ValidationPayload(
        valid=True,
        experiment=experiment_id,
        type=spec.category,
        dependencies=spec.dependencies,
        config_hash=config.config_hash,
    )
    click.echo(json.dumps(payload, indent=2))


@experiment_group.command(name="plan")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option(
    "--experiment",
    type=click.Choice([item.value for item in ExperimentId]),
    default=ExperimentId.PRIMARY_NBAIOT.value,
)
def plan_experiment(config_path: Path | None, experiment: str) -> None:
    """Print the execution plan for one experiment."""
    study = _load_study(config_path)
    experiment_id = ExperimentId(experiment)
    spec = study.spec(experiment_id)
    config = study.resolve(experiment_id)
    model_seeds = config.randomness.model_seeds
    payload = PlanPayload(
        experiment=experiment_id,
        config_hash=config.config_hash,
        model_seed=model_seeds[0] if model_seeds else None,
        calibration_seed=config.dataset.primary_calibration_seed,
        dependencies=spec.dependencies,
    )
    click.echo(json.dumps(payload, indent=2))


_DATASET_EXPERIMENTS: dict[DatasetId, ExperimentId] = {
    DatasetId.NBAIOT: ExperimentId.PRIMARY_NBAIOT,
    DatasetId.DIAD: ExperimentId.EXTERNAL_DIAD,
}


@click.group(name="data")
def data_group() -> None:
    """Prepare seed-independent dataset caches and inspect their state."""


@data_group.command(name="preprocess")
@click.option(
    "--dataset-id",
    type=click.Choice([item.value for item in DatasetId]),
    required=True,
)
@click.option("--data-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
def preprocess_data(dataset_id: str, data_root: Path, config_path: Path | None) -> None:
    """Preprocess one raw dataset into data/preprocessed/ (reuse-first)."""
    dataset = DatasetId(dataset_id)
    try:
        experiment_id = _DATASET_EXPERIMENTS[dataset]
    except KeyError as exc:
        raise click.BadParameter(
            f"Unknown raw dataset id {dataset_id!r}, expected one of "
            + ", ".join(sorted(item.value for item in _DATASET_EXPERIMENTS))
        ) from exc
    study = _load_study(config_path)
    config = study.resolve(experiment_id)
    cache_root = _dataset_preparation().PrepareData().prepare(config, data_root)
    click.echo(str(cache_root))


@data_group.command(name="status")
@click.option(
    "--dataset-id",
    type=click.Choice([item.value for item in DatasetId]),
    default=None,
)
def data_status(dataset_id: str | None) -> None:
    """Show prepared-cache status for one dataset (or all datasets)."""
    entries: list[PreparedEntry] = []
    if _PREPROCESSED_ROOT.is_dir():
        for dataset_root in sorted(path for path in _PREPROCESSED_ROOT.iterdir() if path.is_dir()):
            if dataset_id is not None and dataset_root.name != dataset_id:
                continue
            for identity in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
                manifest = identity / "manifest.json"
                entries.append(
                    PreparedEntry(
                        dataset=DatasetId(dataset_root.name),
                        identity=identity.name,
                        prepared=manifest.is_file(),
                        manifest_sha256=sha256_file(manifest) if manifest.is_file() else None,
                    )
                )
    click.echo(json.dumps(PreparedStatusPayload(prepared=tuple(entries)), indent=2))


@data_group.command(name="prepare")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option("--data-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--experiment",
    type=click.Choice([item.value for item in ExperimentId]),
    default=ExperimentId.PRIMARY_NBAIOT.value,
)
def prepare_data(config_path: Path | None, data_root: Path, experiment: str) -> None:
    """Prepare the seed-independent cache for one experiment from raw data."""
    study = _load_study(config_path)
    config = study.resolve(ExperimentId(experiment))
    cache_root = _dataset_preparation().PrepareData().prepare(config, data_root)
    click.echo(str(cache_root))


@data_group.command(name="prepare-feature-sensitivity")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option("--data-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--eligibility-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Frozen diad_eligibility.json from a prior `data preprocess` run.",
)
@click.option("--feature-manifest", type=click.Path(path_type=Path), required=True)
def prepare_feature_sensitivity(
    config_path: Path | None,
    data_root: Path,
    eligibility_manifest: Path,
    feature_manifest: Path,
) -> None:
    """Freeze the training-schema-only DIAD feature contract and prepare its cache."""
    study = _load_study(config_path)
    config = study.resolve(ExperimentId.DIAD_FEATURE_SENSITIVITY)
    resolved, cache_root = (
        _dataset_preparation()
        .PrepareDiadFeatureSensitivity()
        .prepare(config, data_root, eligibility_manifest, feature_manifest)
    )
    click.echo(f"config_hash={resolved.config_hash}\ncache={cache_root}")


@click.group(name="environment")
def environment_group() -> None:
    """Freeze the validated execution environment before confirmatory runs."""


@environment_group.command(name="freeze")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    show_default=True,
)
@click.option(
    "--repository-root",
    type=click.Path(path_type=Path, exists=True),
    default=Path("."),
    show_default=True,
)
def freeze_environment(output: Path | None, repository_root: Path) -> None:
    """Write the repository and Python environment pin as a JSON document."""
    environment = capture_environment(repository_root)
    target = output or OutputsLayout().environment_file
    atomic_write_json(target, environment)
    payload = FreezePayload(
        path=str(target),
        sha256=environment.environment_pin_sha256,
        git_commit=environment.git_commit,
        git_clean=environment.git_clean,
    )
    click.echo(json.dumps(payload, indent=2))


@click.group(name="tables")
def tables_group() -> None:
    """Precompute protocol tables that are independent of observed client scores."""


@tables_group.command(name="precompute-readiness")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    show_default=True,
)
def precompute_readiness(config_path: Path | None, output: Path | None) -> None:
    """Precompute the per-dataset readiness plan table from protocol constants."""
    study = _load_study(config_path)
    protocol = study.study_config.protocol
    familywise_alpha = study.study_config.statistics.familywise_alpha
    cache = ReadinessPlanCache(builder=ReadinessPlanBuilder())
    for dataset in study.datasets.root.values():
        client_count = (
            dataset.expected_clients
            or dataset.expected_source_clients
            or (len(dataset.expected_benign_counts) if len(dataset.expected_benign_counts) else None)
        )
        if client_count is None:
            continue
        assurance = familywise_readiness_assurance(client_count, familywise_alpha)
        cache.precompute(dataset.split.calibration_benign, protocol.band, assurance)
    target = output or OutputsLayout().readiness_plans_file
    payload = ReadinessTablePayload(plans=cache.plans())
    if target.is_file():
        render_cache_status("readiness", hit=True, target=str(target))
        return
    atomic_write_json(target, payload)
    render_cache_status("readiness", hit=False, target=str(target), detail="plans written")
    click.echo(str(target))


@click.group(name="analysis")
def analysis_group() -> None:
    """Run statistical analyses over frozen evidence."""


@click.group(name="report")
def report_group() -> None:
    """Build reports exclusively from immutable run evidence."""


@report_group.command(name="build")
@click.option("--run", "run_dir", type=click.Path(path_type=Path, exists=True), required=True)
def report_build(run_dir: Path) -> None:
    """Build the report for one completed run."""
    path = _reporting().build_run_report(run_dir)
    click.echo(str(path))


@report_group.command(name="build-repository")
@click.option("--outputs", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option(
    "--experiment",
    type=click.Choice([item.value for item in ExperimentId]),
    default=ExperimentId.PRIMARY_NBAIOT.value,
)
def report_build_repository(
    outputs: Path, config_path: Path | None, experiment: str
) -> None:
    """Build the repository-wide reproducibility index from every completed run."""
    study = _load_study(config_path)
    config = study.resolve(ExperimentId(experiment))
    path = _reporting().build_repository_report(outputs, config)
    click.echo(str(path))


@report_group.command(name="build-publication")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option(
    "--outputs", "outputs_root", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--prepared-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option("--destination", type=click.Path(path_type=Path), default=None)
def report_build_publication(
    config_path: Path | None,
    outputs_root: Path,
    prepared_manifest: Path | None,
    destination: Path | None,
) -> None:
    """Build the manuscript tables and figures from immutable evidence."""
    study = _load_study(config_path)
    config = study.resolve(ExperimentId.PRIMARY_NBAIOT)
    package = _reporting().build_publication(
        config, outputs_root, prepared_manifest, destination
    )
    payload = PublicationPayload(manifest=str(package.manifest), complete=package.complete)
    click.echo(json.dumps(payload, indent=2))


@click.group(name="results")
def results_group() -> None:
    """Build and verify publication bundles under results/<campaign-id>/."""


@results_group.command(name="build")
@click.option("--campaign-id", required=True)
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
@click.option("--results", "results_root", type=click.Path(path_type=Path), default=Path("results"))
def results_build(campaign_id: str, outputs: Path, results_root: Path) -> None:
    """Build the publication bundle for one campaign from immutable evidence."""
    path = _reporting().build_results_bundle(
        _CAMPAIGN_ID.validate_python(campaign_id), outputs_root=outputs, results_root=results_root
    )
    click.echo(json.dumps(ResultsBuildPayload(results_path=str(path)), indent=2))


@results_group.command(name="verify")
@click.option("--campaign-id", required=True)
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
@click.option("--results", "results_root", type=click.Path(path_type=Path), default=Path("results"))
def results_verify(campaign_id: str, outputs: Path, results_root: Path) -> None:
    """Verify that a publication bundle is complete, consistent, and hash-valid."""
    verification = _reporting().verify_results_bundle(
        _CAMPAIGN_ID.validate_python(campaign_id),
        results_root=results_root,
        outputs_root=outputs,
    )
    payload = ResultsVerifyPayload(valid=verification.valid, problems=tuple(verification.problems))
    click.echo(json.dumps(payload, indent=2))
    if not verification.valid:
        raise click.ClickException("Results verification failed")


@click.group(name="sensitivity")
def sensitivity_group() -> None:
    """Run pre-registered real-score sensitivities on a frozen score cache."""


@sensitivity_group.command(name="run")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option("--score-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--experiment",
    type=click.Choice([item.value for item in ExperimentId]),
    required=True,
)
@click.option("--model-seed", type=int, default=None)
@click.option("--calibration-seed", type=int, default=None)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def sensitivity_run(
    config_path: Path | None,
    score_root: Path,
    prepared_root: Path,
    experiment: str,
    model_seed: int | None,
    calibration_seed: int | None,
    output: Path,
) -> None:
    """Run one pre-registered real-score sensitivity on a frozen score cache."""
    study = _load_study(config_path)
    experiment_id = ExperimentId(experiment)
    config = study.resolve(experiment_id)
    typed_model_seed = None if model_seed is None else _MODEL_SEED.validate_python(int(model_seed))
    typed_calibration_seed = (
        None if calibration_seed is None else _CALIBRATION_SEED.validate_python(int(calibration_seed))
    )
    path = _sensitivity_runner().run(
        experiment_id,
        config,
        score_root,
        prepared_root,
        model_seed=typed_model_seed,
        calibration_seed=typed_calibration_seed,
        output=output,
    )
    click.echo(str(path))


@click.group(name="robustness")
def robustness_group() -> None:
    """Run mandatory outcome-independent robustness checks."""


@robustness_group.command(name="deep-svdd")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--model-seed", type=int, required=True)
def train_deep_svdd(
    config_path: Path | None, prepared_root: Path, model_seed: int
) -> None:
    """Train the mandatory outcome-independent Deep-SVDD second score generator."""
    study = _load_study(config_path)
    config = study.resolve(ExperimentId.SECOND_DETECTOR)
    model, manifest = _robustness_runner().run_deep_svdd(
        config, prepared_root, _MODEL_SEED.validate_python(int(model_seed))
    )
    click.echo(f"model={model}\nmanifest={manifest}")


@click.group(name="synthetic")
def synthetic_group() -> None:
    """Run pre-registered synthetic validation cells."""


@synthetic_group.command(name="run")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option(
    "--experiment",
    type=click.Choice([item.value for item in ExperimentId]),
    required=True,
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def synthetic_run(
    config_path: Path | None, experiment: str, output: Path
) -> None:
    """Run one pre-registered synthetic validation cell."""
    study = _load_study(config_path)
    experiment_id = ExperimentId(experiment)
    config = study.resolve(experiment_id)
    path = _synthetic_runner().run(experiment_id, config, output)
    click.echo(str(path))


@click.group(name="campaign")
def campaign_group() -> None:
    """Run and inspect persistent research campaigns."""


@campaign_group.command(name="run")
@click.option("--campaign-id", required=True)
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option(
    "--experiment",
    type=click.Choice([item.value for item in ExperimentId]),
    multiple=True,
)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
@click.option("--results", "results_root", type=click.Path(path_type=Path), default=Path("results"))
def campaign_run(
    campaign_id: str,
    config_path: Path | None,
    experiment: tuple[str, ...],
    prepared_root: Path,
    outputs: Path,
    results_root: Path,
) -> None:
    """Execute a campaign over the given experiment configs and record status."""
    module = _campaign_module()
    typed_campaign_id = _CAMPAIGN_ID.validate_python(campaign_id)
    experiment_ids = (
        tuple(ExperimentId(item) for item in experiment) if experiment else tuple(ExperimentId)
    )
    resolved_config = config_path if config_path is not None else _DEFAULT_STUDY_CONFIG
    work_items = tuple(
        module.CampaignWorkItem(
            experiment_id=experiment_id,
            config_path=resolved_config,
            prepared_root=prepared_root,
        )
        for experiment_id in experiment_ids
    )
    status = module.CampaignRunner().run(
        typed_campaign_id,
        work_items,
        outputs_root=outputs,
        results_root=results_root,
    )
    payload = CampaignStatusPayload(
        campaign_id=typed_campaign_id,
        status=status.status,
        completed=status.completed,
        total=status.total,
        current_experiment=(
            ExperimentId(status.current_experiment) if status.current_experiment is not None else None
        ),
        elapsed_seconds=int(status.elapsed_seconds),
    )
    click.echo(json.dumps(payload, indent=2))


@campaign_group.command(name="status")
@click.option("--campaign-id", required=True)
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
def campaign_status(campaign_id: str, outputs: Path) -> None:
    """Show persistent status for one campaign."""
    typed_campaign_id = _CAMPAIGN_ID.validate_python(campaign_id)
    status = _campaign_module().CampaignStatusStore(
        outputs_root=outputs
    ).load(typed_campaign_id)
    payload = CampaignStatusPayload(
        campaign_id=typed_campaign_id,
        status=status.status,
        completed=status.completed,
        total=status.total,
        current_experiment=(
            ExperimentId(status.current_experiment) if status.current_experiment is not None else None
        ),
        elapsed_seconds=int(status.elapsed_seconds),
    )
    click.echo(json.dumps(payload, indent=2))


@campaign_group.command(name="report")
@click.option("--campaign-id", required=True)
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
def campaign_report(campaign_id: str, outputs: Path) -> None:
    """Render a status table for one campaign."""
    typed_campaign_id = _CAMPAIGN_ID.validate_python(campaign_id)
    status = _campaign_module().CampaignStatusStore(
        outputs_root=outputs
    ).load(typed_campaign_id)
    render_campaign_status(
        typed_campaign_id,
        status.status,
        status.completed,
        status.total,
        status.current_experiment,
        status.elapsed_seconds,
    )


@campaign_group.command(name="list")
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
def campaign_list(outputs: Path) -> None:
    """List recorded campaign ids."""
    campaigns_root = OutputsLayout(outputs).campaigns
    ids = (
        sorted(path.stem for path in campaigns_root.glob("*.json"))
        if campaigns_root.is_dir()
        else []
    )
    click.echo(json.dumps(ids, indent=2))


cli.add_command(experiment_group)
cli.add_command(data_group)
cli.add_command(environment_group)
cli.add_command(tables_group)
cli.add_command(analysis_group)
cli.add_command(report_group)
cli.add_command(results_group)
cli.add_command(sensitivity_group)
cli.add_command(robustness_group)
cli.add_command(synthetic_group)
cli.add_command(campaign_group)


if __name__ == "__main__":
    cli()
