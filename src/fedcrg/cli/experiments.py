"""Experiment planning CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.cli.shared import load_config
from fedcrg.core.enums import ExperimentId, PolicyId
from fedcrg.experiments.planner import ExperimentPlanner


@click.group(name="experiment")
def experiment_group() -> None:
    """Plan typed experiment executions."""


@experiment_group.command(name="plan")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--experiment", type=click.Choice([item.value for item in ExperimentId]), default=None)
def plan_experiment(config_path: Path, experiment: str | None) -> None:
    config = load_config(config_path)
    experiment_id = ExperimentId(experiment) if experiment is not None else config.id
    plan = ExperimentPlanner().create(
        experiment_id,
        config,
        model_seed=config.randomness.model_seeds[0],
        calibration_seed=config.dataset.primary_calibration_seed,
    )
    click.echo(json.dumps({
        "experiment": plan.definition.id.value,
        "config_hash": plan.config_hash.value,
        "model_seed": plan.model_seed,
        "calibration_seed": plan.calibration_seed,
        "dependencies": [item.value for item in plan.definition.dependencies],
    }, indent=2))


@experiment_group.command(name="run-policy-cell")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--experiment", type=click.Choice([item.value for item in ExperimentId]), default=None)
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
    from fedcrg.application.policy_cell import FrozenCacheInputs, PolicyCellMaterializer
    from fedcrg.application.run_experiment import RunExperiment
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
