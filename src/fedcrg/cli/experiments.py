"""Experiment planning CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.cli.shared import load_config
from fedcrg.core.enums import ExperimentId
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
        "config_hash": plan.config_hash,
        "model_seed": plan.model_seed,
        "calibration_seed": plan.calibration_seed,
        "dependencies": [item.value for item in plan.definition.dependencies],
    }, indent=2))
