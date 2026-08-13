"""Report, results-bundle, and campaign-status CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.configuration.resolve import load_config
from fedcrg.reporting.report import ReportBuilder
from fedcrg.reporting.results import ResultsBuilder, ResultsVerifier


@click.group(name="report")
def report_group() -> None:
    """Build reports exclusively from immutable run evidence."""


@report_group.command(name="build")
@click.option("--run", "run_dir", type=click.Path(path_type=Path, exists=True), required=True)
def report_build(run_dir: Path) -> None:
    click.echo(str(ReportBuilder().build(run_dir)))


@report_group.command(name="build-repository")
@click.option("--outputs", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
def report_build_repository(outputs: Path, config_path: Path) -> None:
    """Build the repository-wide reproducibility index from every completed run."""
    click.echo(str(ReportBuilder().build_repository(outputs, load_config(config_path))))


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


@click.group(name="results")
def results_group() -> None:
    """Build and verify publication bundles under results/<campaign_id>/."""


@results_group.command(name="build")
@click.option("--campaign-id", required=True)
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
@click.option("--results", "results_root", type=click.Path(path_type=Path), default=Path("results"))
def results_build(campaign_id: str, outputs: Path, results_root: Path) -> None:
    """Build the publication bundle for one campaign from immutable evidence."""
    path = ResultsBuilder().build(
        campaign_id=campaign_id,
        outputs_root=outputs,
        results_root=results_root,
    )
    click.echo(json.dumps({"results_path": str(path)}, indent=2))


@results_group.command(name="verify")
@click.option("--campaign-id", required=True)
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
@click.option("--results", "results_root", type=click.Path(path_type=Path), default=Path("results"))
def results_verify(campaign_id: str, outputs: Path, results_root: Path) -> None:
    """Verify that a publication bundle is complete, consistent, and hash-valid."""
    result = ResultsVerifier().verify(
        campaign_id,
        results_root=results_root,
        outputs_root=outputs,
    )
    payload = {"valid": result.valid, "problems": list(result.problems)}
    click.echo(json.dumps(payload, indent=2))
    if not result.valid:
        raise click.ClickException("Results verification failed")
