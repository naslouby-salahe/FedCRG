"""Experiment, training, scoring, sensitivity, robustness, synthetic, benchmark,
and campaign CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.configuration.resolve import load_config
from fedcrg.domain.enums import DetectorId, ExperimentId, PolicyId
from fedcrg.experiments.planning import ExperimentPlanner


@click.group(name="experiment")
def experiment_group() -> None:
    """Plan typed experiment executions."""


@experiment_group.command(name="validate")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--experiment", type=click.Choice([item.value for item in ExperimentId]), default=None
)
def validate_experiment(config_path: Path, experiment: str | None) -> None:
    """Validate a fully resolved experiment configuration against the catalogue."""
    config = load_config(config_path)
    experiment_id = ExperimentId(experiment) if experiment is not None else config.id
    if experiment_id is not config.id:
        raise click.ClickException(
            f"Config id {config.id.value} does not match requested {experiment_id.value}"
        )
    definition = (
        ExperimentPlanner()
        .create(
            experiment_id,
            config,
            model_seed=config.randomness.model_seeds[0],
            calibration_seed=config.dataset.primary_calibration_seed,
        )
        .definition
    )
    click.echo(
        json.dumps(
            {
                "valid": True,
                "experiment": experiment_id.value,
                "type": definition.type.value,
                "dependencies": [item.value for item in definition.dependencies],
                "config_hash": config.config_hash,
            },
            indent=2,
        )
    )


@experiment_group.command(name="plan")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--experiment", type=click.Choice([item.value for item in ExperimentId]), default=None
)
def plan_experiment(config_path: Path, experiment: str | None) -> None:
    config = load_config(config_path)
    experiment_id = ExperimentId(experiment) if experiment is not None else config.id
    plan = ExperimentPlanner().create(
        experiment_id,
        config,
        model_seed=config.randomness.model_seeds[0],
        calibration_seed=config.dataset.primary_calibration_seed,
    )
    click.echo(
        json.dumps(
            {
                "experiment": plan.definition.id.value,
                "config_hash": plan.config_hash.value,
                "model_seed": plan.model_seed,
                "calibration_seed": plan.calibration_seed,
                "dependencies": [item.value for item in plan.definition.dependencies],
            },
            indent=2,
        )
    )


@experiment_group.command(name="run-policy-cell")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--experiment", type=click.Choice([item.value for item in ExperimentId]), default=None
)
@click.option("--policy", type=click.Choice([item.value for item in PolicyId]), required=True)
@click.option("--model-seed", type=int, required=True)
@click.option("--calibration-seed", type=int, required=True)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--model-path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--training-manifest", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--score-root", type=click.Path(path_type=Path, exists=True), required=True)
def run_policy_cell(
    config_path: Path,
    experiment: str | None,
    policy: str,
    model_seed: int,
    calibration_seed: int,
    prepared_root: Path,
    model_path: Path,
    training_manifest: Path,
    score_root: Path,
) -> None:
    """Materialize one immutable pre-registered policy cell from frozen caches."""
    from fedcrg.experiments.experiment_runner import RunExperiment
    from fedcrg.experiments.policy_cells import FrozenCacheInputs, PolicyCellMaterializer

    config = load_config(config_path)
    experiment_id = ExperimentId(experiment) if experiment is not None else config.id
    policy_id = PolicyId(policy)
    caches = FrozenCacheInputs(prepared_root, model_path, training_manifest, score_root)
    materializer = PolicyCellMaterializer()
    _, layout = RunExperiment().execute(
        experiment_id=experiment_id,
        config=config,
        model_seed=model_seed,
        calibration_seed=calibration_seed,
        policy=policy_id,
        runner=lambda _plan, run_layout: materializer.materialize(
            config, policy_id, run_layout, caches, calibration_seed
        ),
    )
    click.echo(str(layout.root))


@experiment_group.command(name="materialize-federation-cell")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--experiment", type=click.Choice([item.value for item in ExperimentId]), default=None
)
@click.option("--model-seed", type=int, required=True)
@click.option("--calibration-seed", type=int, required=True)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--model-path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--training-manifest", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--score-root", type=click.Path(path_type=Path, exists=True), required=True)
def materialize_federation_cell(
    config_path: Path,
    experiment: str | None,
    model_seed: int,
    calibration_seed: int,
    prepared_root: Path,
    model_path: Path,
    training_manifest: Path,
    score_root: Path,
) -> None:
    """Evaluate one federation once and materialize every configured policy run."""
    from fedcrg.experiments.policy_cells import FederationCellMaterializer, FrozenCacheInputs

    config = load_config(config_path)
    experiment_id = ExperimentId(experiment) if experiment is not None else config.id
    result = FederationCellMaterializer().materialize(
        experiment_id,
        config,
        model_seed,
        calibration_seed,
        FrozenCacheInputs(prepared_root, model_path, training_manifest, score_root),
    )
    click.echo(
        json.dumps(
            {entry.policy.value: str(entry.path) for entry in result.run_directories},
            indent=2,
        )
    )


@experiment_group.command(name="execute-grid")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--experiment", type=click.Choice([item.value for item in ExperimentId]), default=None
)
@click.option("--named-only", is_flag=True, default=False)
def execute_grid(
    config_path: Path,
    prepared_root: Path,
    experiment: str | None,
    named_only: bool,
) -> None:
    """Audit prepared evidence, then execute each model seed once, score once, and
    materialize the requested policy grid."""
    from fedcrg.experiments.runner import RunAllExperiments

    config = load_config(config_path)
    experiment_id = ExperimentId(experiment) if experiment is not None else config.id
    calibration_seeds = (
        (config.dataset.primary_calibration_seed,)
        if named_only
        else config.dataset.calibration_seeds
    )
    execution = RunAllExperiments().execute(
        experiment_id,
        config,
        prepared_root,
        calibration_seeds=calibration_seeds,
    )
    result = execution.workload
    click.echo(
        json.dumps(
            {
                "experiment": experiment_id.value,
                "model_seeds": [item.model_seed for item in result.models],
                "model_count": len(result.models),
                "policy_run_count": len(result.run_directories),
                "calibration_seeds": list(calibration_seeds),
                "preflight_client_count": execution.preflight.prepared_data.client_count,
            },
            indent=2,
        )
    )


@click.command(name="train")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--model-seed", type=int, required=True)
def train_command(config_path: Path, prepared_root: Path, model_seed: int) -> None:
    from fedcrg.experiments.model_training import TrainDetector

    model_path, manifest_path = TrainDetector().train_from_cache(
        load_config(config_path), prepared_root, model_seed
    )
    click.echo(json.dumps({"model": str(model_path), "manifest": str(manifest_path)}, indent=2))


@click.command(name="score")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--model", "model_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--training-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Frozen training manifest that proves the model's scientific provenance.",
)
@click.option("--model-seed", type=int, required=True)
def score_command(
    config_path: Path,
    prepared_root: Path,
    model_path: Path,
    training_manifest: Path,
    model_seed: int,
) -> None:
    from fedcrg.scoring.compute_scores import ComputeScores

    score_root = ComputeScores().score_from_cache(
        load_config(config_path),
        prepared_root,
        model_path,
        model_seed,
        training_manifest,
    )
    click.echo(str(score_root))


@click.command(name="evaluate")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--score-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--calibration-seed", type=int, default=None)
@click.option("--output", type=click.Path(path_type=Path), default=Path("outputs/evaluation.json"))
def evaluate_command(
    config_path: Path, score_root: Path, calibration_seed: int | None, output: Path
) -> None:
    from fedcrg.artifacts.json_io import atomic_write_json
    from fedcrg.experiments.policy_evaluation import EvaluatePolicies

    config = load_config(config_path)
    service = EvaluatePolicies()
    bundle = service.evaluate_from_cache(config, score_root, calibration_seed=calibration_seed)
    atomic_write_json(output, service.to_serializable(bundle))
    click.echo(str(output))


@click.command(name="benchmark")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("outputs/reports/latest/benchmark.json"),
)
def benchmark_command(config_path: Path, output: Path) -> None:
    from fedcrg.experiments.computational_benchmark import RunBenchmark

    config = load_config(config_path)
    path = RunBenchmark().run_on_synthetic_evidence(config, output)
    click.echo(str(path))


_SYNTHETIC_EXPERIMENTS = (
    ExperimentId.READINESS_THEOREM,
    ExperimentId.TARGET_FPR_SYNTHETIC,
    ExperimentId.TEMPORAL_DEPENDENCE,
    ExperimentId.CALIBRATION_SHIFT,
    ExperimentId.CALIBRATION_CONTAMINATION,
    ExperimentId.MISMATCH_POWER,
)


@click.group(name="synthetic")
def synthetic_group() -> None:
    """Run pre-registered synthetic validation cells."""


@synthetic_group.command(name="run")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--experiment",
    type=click.Choice([item.value for item in _SYNTHETIC_EXPERIMENTS]),
    required=True,
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def synthetic_run(config_path: Path, experiment: str, output: Path) -> None:
    from fedcrg.experiments.definitions.synthetic import RunSyntheticExperiments

    config = load_config(config_path)
    experiment_id = ExperimentId(experiment)
    runner = RunSyntheticExperiments()
    if experiment_id is ExperimentId.READINESS_THEOREM:
        path = runner.run_s1(config, output)
    elif experiment_id is ExperimentId.TARGET_FPR_SYNTHETIC:
        path = runner.run_s2(config, output)
    elif experiment_id is ExperimentId.TEMPORAL_DEPENDENCE:
        path = runner.run_s3(config, output)
    elif experiment_id is ExperimentId.CALIBRATION_SHIFT:
        path = runner.run_s4(config, output)
    elif experiment_id is ExperimentId.CALIBRATION_CONTAMINATION:
        path = runner.run_s5(config, output)
    else:
        path = runner.run_s6(output)
    click.echo(str(path))


@click.group(name="robustness")
def robustness_group() -> None:
    """Run mandatory outcome-independent robustness checks."""


@robustness_group.command(name="deep-svdd")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--model-seed", type=int, required=True)
def train_deep_svdd(config_path: Path, prepared_root: Path, model_seed: int) -> None:
    """Train the mandatory outcome-independent Deep-SVDD second score generator."""
    from fedcrg.experiments.model_training import TrainDetector

    config = load_config(config_path)
    if config.detector.id is not DetectorId.DEEP_SVDD:
        raise ValueError("Second-detector robustness requires the Deep-SVDD config")
    model, manifest = TrainDetector().train_from_cache(config, prepared_root, model_seed)
    click.echo(f"model={model}\nmanifest={manifest}")


_MODEL_SEED_SENSITIVITY_EXPERIMENTS = (
    ExperimentId.READINESS_SAMPLE_SIZE,
    ExperimentId.MISMATCH_SAMPLE_SIZE,
    ExperimentId.TOLERANCE_SENSITIVITY,
    ExperimentId.TARGET_FPR_REAL,
    ExperimentId.ASSURANCE_SENSITIVITY,
)
_SINGLE_SEED_SENSITIVITY_EXPERIMENTS = (
    ExperimentId.MULTIPLICITY_SENSITIVITY,
    ExperimentId.SOURCE_ORDER_TEST,
    ExperimentId.REAL_CONTAMINATION,
)
_SENSITIVITY_EXPERIMENTS = (
    *_MODEL_SEED_SENSITIVITY_EXPERIMENTS,
    *_SINGLE_SEED_SENSITIVITY_EXPERIMENTS,
    ExperimentId.SOURCE_ORDER_CALIBRATION,
)


@click.group(name="sensitivity")
def sensitivity_group() -> None:
    """Run pre-registered real-score sensitivities on a frozen score cache."""


@sensitivity_group.command(name="run")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--score-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--experiment",
    type=click.Choice([item.value for item in _SENSITIVITY_EXPERIMENTS]),
    required=True,
)
@click.option("--model-seed", type=int, default=None)
@click.option("--calibration-seed", type=int, default=None)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def sensitivity_run(
    config_path: Path,
    score_root: Path,
    prepared_root: Path,
    experiment: str,
    model_seed: int | None,
    calibration_seed: int | None,
    output: Path,
) -> None:
    """Run one pre-registered real-score sensitivity on a frozen score cache."""
    from fedcrg.experiments.definitions.sensitivity import (
        RunRealSensitivities,
        RunSourceOrderCalibration,
    )

    config = load_config(config_path)
    experiment_id = ExperimentId(experiment)

    if experiment_id is ExperimentId.SOURCE_ORDER_CALIBRATION:
        path = RunSourceOrderCalibration().run(config, prepared_root, score_root, output)
        click.echo(str(path))
        return

    runner = RunRealSensitivities()
    model_seed_methods = {
        ExperimentId.READINESS_SAMPLE_SIZE: runner.run_r2,
        ExperimentId.MISMATCH_SAMPLE_SIZE: runner.run_r3,
        ExperimentId.TOLERANCE_SENSITIVITY: runner.run_r4,
        ExperimentId.TARGET_FPR_REAL: runner.run_r5,
        ExperimentId.ASSURANCE_SENSITIVITY: runner.run_r6,
    }
    single_seed_methods = {
        ExperimentId.MULTIPLICITY_SENSITIVITY: runner.run_r7,
        ExperimentId.SOURCE_ORDER_TEST: runner.run_r8,
        ExperimentId.REAL_CONTAMINATION: runner.run_r9,
    }
    if experiment_id in model_seed_methods:
        if model_seed is None:
            raise click.UsageError(f"{experiment} requires --model-seed")
        path = model_seed_methods[experiment_id](
            config, score_root, prepared_root, model_seed, output, calibration_seed
        )
    else:
        path = single_seed_methods[experiment_id](
            config, score_root, prepared_root, output, calibration_seed
        )
    click.echo(str(path))


@click.group(name="campaign")
def campaign_group() -> None:
    """Run and inspect persistent research campaigns."""


@campaign_group.command(name="run")
@click.option("--campaign-id", required=True)
@click.option(
    "--config", "config_paths", type=click.Path(path_type=Path, exists=True), multiple=True
)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
@click.option("--results", "results_root", type=click.Path(path_type=Path), default=Path("results"))
def campaign_run(
    campaign_id: str,
    config_paths: tuple[Path, ...],
    prepared_root: Path,
    outputs: Path,
    results_root: Path,
) -> None:
    """Execute a campaign over the given experiment configs and record persistent status."""
    from fedcrg.experiments.campaign import CampaignRunner, CampaignWorkItem

    if not config_paths:
        raise click.UsageError("At least one --config is required")
    work_items = tuple(
        CampaignWorkItem(
            experiment_id=load_config(path).id,
            config_path=path,
            prepared_root=prepared_root,
        )
        for path in config_paths
    )
    status = CampaignRunner().run(
        campaign_id,
        work_items,
        outputs_root=outputs,
        results_root=results_root,
    )
    click.echo(json.dumps(status.to_dict(), indent=2))


@campaign_group.command(name="status")
@click.option("--campaign-id", required=True)
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
def campaign_status(campaign_id: str, outputs: Path) -> None:
    """Show persistent status for one campaign."""
    from fedcrg.experiments.campaign import CampaignStatusStore

    status = CampaignStatusStore(outputs / "campaigns").load(campaign_id)
    click.echo(json.dumps(status.to_dict(), indent=2))


@campaign_group.command(name="list")
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
def campaign_list(outputs: Path) -> None:
    """List recorded campaigns."""
    campaigns_root = outputs / "campaigns"
    if not campaigns_root.exists():
        click.echo("[]")
        return
    ids = sorted(path.stem for path in campaigns_root.glob("*.json"))
    click.echo(json.dumps(ids, indent=2))
