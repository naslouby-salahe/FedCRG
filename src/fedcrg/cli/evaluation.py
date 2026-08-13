"""Evaluation and report CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.pipeline.evaluate_policies import EvaluatePolicies
from fedcrg.reporting.report import ReportBuilder
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


@report_group.command(name="build-repository")
@click.option("--outputs", type=click.Path(path_type=Path, exists=True), required=True)
def report_build_repository(outputs: Path) -> None:
    """Build the repository-wide reproducibility index from every completed run."""
    click.echo(str(ReportBuilder().build_repository(outputs)))


@report_group.command(name="build-publication")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--outputs", "outputs_root", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--prepared-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option("--destination", type=click.Path(path_type=Path), default=None)
def report_build_publication(
    config_path: Path,
    outputs_root: Path,
    prepared_manifest: Path | None,
    destination: Path | None,
) -> None:
    """Build the manuscript Tables 1-8 and Figures 1-8 from immutable evidence."""
    from fedcrg.reporting.publication import PublicationPackageBuilder

    package = PublicationPackageBuilder().build(
        config=load_config(config_path),
        outputs_root=outputs_root,
        prepared_manifest=prepared_manifest,
        destination=destination,
    )
    click.echo(
        json.dumps({"manifest": str(package.manifest), "complete": package.complete}, indent=2)
    )
