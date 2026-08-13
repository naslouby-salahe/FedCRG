"""Detector training CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.experiments.model_training import TrainDetector
from fedcrg.configuration.resolve import load_config


@click.command(name="train")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--model-seed", type=int, required=True)
def train_command(config_path: Path, prepared_root: Path, model_seed: int) -> None:
    model_path, manifest_path = TrainDetector().train_from_cache(
        load_config(config_path), prepared_root, model_seed
    )
    click.echo(json.dumps({"model": str(model_path), "manifest": str(manifest_path)}, indent=2))
