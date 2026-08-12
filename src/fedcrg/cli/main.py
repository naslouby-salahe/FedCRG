"""FedCRG command-line entry point."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.cli.data import data_group
from fedcrg.cli.evaluation import evaluate_command, report_command
from fedcrg.cli.experiments import experiment_group
from fedcrg.cli.shared import load_config
from fedcrg.cli.training import score_command, train_command
from fedcrg.cli.verification import verify_command


@click.group()
@click.version_option(package_name="fedcrg")
def cli() -> None:
    """FedCRG reproducible experiment tooling."""


@cli.group(name="config")
def config_group() -> None:
    """Validate resolved experiment configurations."""


@config_group.command(name="validate")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
def validate_config(config_path: Path) -> None:
    config = load_config(config_path)
    click.echo(json.dumps({"valid": True, "config_hash": config.config_hash}, indent=2))


cli.add_command(data_group)
cli.add_command(train_command)
cli.add_command(score_command)
cli.add_command(evaluate_command)
cli.add_command(report_command)
cli.add_command(experiment_group)
cli.add_command(verify_command)


if __name__ == "__main__":
    cli()
