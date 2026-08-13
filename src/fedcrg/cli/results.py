"""Publication results build/verify CLI."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.reporting.results import ResultsBuilder, ResultsVerifier


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
