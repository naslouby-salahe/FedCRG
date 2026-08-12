"""Evaluation and reporting CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from fedcrg.application.evaluate import EvaluatePolicies
from fedcrg.application.report import ReportBuilder
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.cli.shared import load_config
from fedcrg.scoring.cache import ScoreCache


@click.command(name="evaluate")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--score-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("outputs/evaluation.json"))
def evaluate_command(config_path: Path, score_root: Path, output: Path) -> None:
    config = load_config(config_path)
    service = EvaluatePolicies()
    evaluations = service.evaluate(config, ScoreCache().load(score_root))
    atomic_write_json(output, service.to_serializable(evaluations))
    click.echo(str(output))


@click.command(name="report")
@click.option("--run", "run_dir", type=click.Path(path_type=Path, exists=True), required=True)
def report_command(run_dir: Path) -> None:
    click.echo(str(ReportBuilder().build(run_dir)))
