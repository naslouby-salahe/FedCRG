"""Evaluation and report CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from fedcrg.application.evaluate import EvaluatePolicies
from fedcrg.application.report import ReportBuilder
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.cli.shared import load_config


@click.command(name="evaluate")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--score-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--calibration-seed", type=int, default=None)
@click.option("--output", type=click.Path(path_type=Path), default=Path("outputs/evaluation.json"))
def evaluate_command(
    config_path: Path, score_root: Path, calibration_seed: int | None, output: Path
) -> None:
    config = load_config(config_path)
    service = EvaluatePolicies()
    bundle = service.evaluate_from_cache(config, score_root, calibration_seed=calibration_seed)
    atomic_write_json(output, service.to_serializable(bundle))
    click.echo(str(output))


@click.group(name="report")
def report_group() -> None:
    """Build reports exclusively from immutable run evidence."""


@report_group.command(name="build")
@click.option("--run", "run_dir", type=click.Path(path_type=Path, exists=True), required=True)
def report_build(run_dir: Path) -> None:
    click.echo(str(ReportBuilder().build(run_dir)))
