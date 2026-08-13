"""Experiment planning CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.cli.shared import load_config
from fedcrg.domain.enums import ExperimentId, PolicyId
from fedcrg.experiments.planning import ExperimentPlanner


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
