"""Experiment planning and execution CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.domain.enums import DetectorId, ExperimentId, PolicyId
from fedcrg.experiments.definitions.synthetic import RunSyntheticExperiments
from fedcrg.experiments.planning import ExperimentPlanner
from fedcrg.pipeline.train_detector import TrainDetector
from fedcrg.runtime import load_config


@click.group(name="experiment")
def experiment_group() -> None:
    """Plan typed experiment executions."""


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
    from fedcrg.pipeline.run_experiment import RunExperiment
    from fedcrg.pipeline.run_policy_evaluation import FrozenCacheInputs, PolicyCellMaterializer

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
    from fedcrg.pipeline.run_policy_evaluation import FederationCellMaterializer, FrozenCacheInputs

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
    from fedcrg.pipeline.run_all_experiments import RunAllExperiments

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


@click.group(name="synthetic")
def synthetic_group() -> None:
    """Run pre-registered synthetic validation cells."""


@synthetic_group.command(name="run")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--experiment", type=click.Choice(["S1", "S2", "S3", "S4", "S5", "S6"]), required=True
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def synthetic_run(config_path: Path, experiment: str, output: Path) -> None:
    config = load_config(config_path)
    runner = RunSyntheticExperiments()
    if experiment == "S1":
        path = runner.run_s1(config, output)
    elif experiment == "S2":
        path = runner.run_s2(config, output)
    elif experiment == "S3":
        path = runner.run_s3(config, output)
    elif experiment == "S4":
        path = runner.run_s4(config, output)
    elif experiment == "S5":
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
    config = load_config(config_path)
    if config.detector.id is not DetectorId.DEEP_SVDD:
        raise ValueError("Second-detector robustness requires the Deep-SVDD config")
    model, manifest = TrainDetector().train_from_cache(config, prepared_root, model_seed)
    click.echo(f"model={model}\nmanifest={manifest}")


@click.group(name="sensitivity")
def sensitivity_group() -> None:
    """Run pre-registered real-score sensitivities on a frozen score cache."""


_MODEL_SEED_SENSITIVITIES = frozenset({"R2", "R3", "R4", "R5", "R6"})


@sensitivity_group.command(name="run")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--score-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--experiment",
    type=click.Choice(["R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R12"]),
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
    """Run one pre-registered R2-R9/R12 real-score sensitivity on a frozen score cache."""
    from fedcrg.experiments.definitions.sensitivity import (
        RunRealSensitivities,
        RunSourceOrderCalibration,
    )

    config = load_config(config_path)

    if experiment == "R12":
        path = RunSourceOrderCalibration().run(config, prepared_root, score_root, output)
        click.echo(str(path))
        return

    runner = RunRealSensitivities()
    methods = {
        "R2": runner.run_r2,
        "R3": runner.run_r3,
        "R4": runner.run_r4,
        "R5": runner.run_r5,
        "R6": runner.run_r6,
        "R7": runner.run_r7,
        "R8": runner.run_r8,
        "R9": runner.run_r9,
    }
    method = methods[experiment]
    if experiment in _MODEL_SEED_SENSITIVITIES:
        if model_seed is None:
            raise click.UsageError(f"{experiment} requires --model-seed")
        path = method(config, score_root, prepared_root, model_seed, output, calibration_seed)
    else:
        path = method(config, score_root, prepared_root, output, calibration_seed)
    click.echo(str(path))
