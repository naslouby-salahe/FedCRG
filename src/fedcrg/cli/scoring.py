"""Scoring and pre-data statistical table CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from fedcrg.scoring.compute_scores import ComputeScores
from fedcrg.experiments.table_precompute import ProtocolTablePrecomputer
from fedcrg.configuration.resolve import load_config


@click.group(name="tables")
def tables_group() -> None:
    """Precompute protocol tables that are independent of observed client scores."""


@tables_group.command(name="precompute-readiness")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("outputs/cache/analysis/readiness_plans.json"),
)
def precompute_readiness(config_path: Path, output: Path) -> None:
    config = load_config(config_path)
    readiness_path, _mismatch_path = ProtocolTablePrecomputer().precompute(config, output.parent)
    click.echo(str(readiness_path))


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
    score_root = ComputeScores().score_from_cache(
        load_config(config_path),
        prepared_root,
        model_path,
        model_seed,
        training_manifest,
    )
    click.echo(str(score_root))
