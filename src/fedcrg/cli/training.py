"""Training and scoring CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.application.score import ComputeScores
from fedcrg.application.train import TrainDetector
from fedcrg.cli.shared import load_config


@click.command(name="train")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--model-seed", type=int, required=True)
def train_command(config_path: Path, prepared_root: Path, model_seed: int) -> None:
    model_path, manifest_path = TrainDetector().train_from_cache(
        load_config(config_path), prepared_root, model_seed
    )
    click.echo(json.dumps({"model": str(model_path), "manifest": str(manifest_path)}, indent=2))


@click.command(name="score")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
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
    score_root = ComputeScores().score_from_cache(
        load_config(config_path),
        prepared_root,
        model_path,
        model_seed,
        training_manifest,
    )
    click.echo(str(score_root))
