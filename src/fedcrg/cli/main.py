"""FedCRG research command-line entry point."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import click
import numpy
import pandas
import scipy
import torch

from fedcrg.cli.benchmark import benchmark_command
from fedcrg.cli.claims import claims_group
from fedcrg.cli.data import data_group
from fedcrg.cli.environment import environment_group
from fedcrg.cli.evaluation import evaluate_command
from fedcrg.cli.experiments import (
    experiment_group,
    robustness_group,
    sensitivity_group,
    synthetic_group,
)
from fedcrg.cli.reporting import report_group
from fedcrg.cli.scoring import score_command, tables_group
from fedcrg.cli.training import train_command
from fedcrg.cli.verification import verify_command
from fedcrg.runtime import configure_logging, load_config


@click.group()
@click.version_option(package_name="fedcrg")
def cli() -> None:
    """FedCRG reproducible research tooling."""
    configure_logging()


@cli.command(name="doctor")
def doctor() -> None:
    click.echo(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "pandas": pandas.__version__,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            indent=2,
        )
    )


@cli.group(name="config")
def config_group() -> None:
    """Validate fully resolved experiment configurations."""


@config_group.command(name="validate")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
def validate_config(config_path: Path) -> None:
    config = load_config(config_path)
    click.echo(json.dumps({"valid": True, "config_hash": config.config_hash}, indent=2))


cli.add_command(claims_group)
cli.add_command(data_group)
cli.add_command(environment_group)
cli.add_command(tables_group)
cli.add_command(synthetic_group)
cli.add_command(train_command)
cli.add_command(score_command)
cli.add_command(evaluate_command)
cli.add_command(robustness_group)
cli.add_command(sensitivity_group)
cli.add_command(benchmark_command)
cli.add_command(report_group)
cli.add_command(experiment_group)
cli.add_command(verify_command)

if __name__ == "__main__":
    cli()
