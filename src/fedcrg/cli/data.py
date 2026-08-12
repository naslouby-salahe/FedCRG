"""Dataset CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from fedcrg.application.prepare_data import PrepareData
from fedcrg.cli.shared import load_config


@click.group(name="data")
def data_group() -> None:
    """Prepare and validate dataset caches."""


@data_group.command(name="prepare")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--data-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--calibration-seed", type=int, default=None)
def prepare_data(config_path: Path, data_root: Path, calibration_seed: int | None) -> None:
    cache_root = PrepareData().prepare(load_config(config_path), data_root, calibration_seed)
    click.echo(str(cache_root))
