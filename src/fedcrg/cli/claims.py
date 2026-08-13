"""Claim-strength gate (G0-G8) CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from fedcrg.analysis.claim_gates import ClaimGateEvaluator


@click.group(name="claims")
def claims_group() -> None:
    """Evaluate the pre-registered claim-strength gates from frozen evidence."""


@claims_group.command(name="evaluate")
@click.option("--outputs", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--novelty-log", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None
)
@click.option("--repository-certified", is_flag=True, default=False)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def claims_evaluate(
    outputs: Path,
    novelty_log: Path | None,
    repository_certified: bool,
    output: Path | None,
) -> None:
    """Write the G0-G8 claim-strength assessment. Never redesigns on the outcome."""
    path = ClaimGateEvaluator().write(
        outputs,
        output or outputs / "reports" / "latest" / "claims.json",
        novelty_log=novelty_log,
        repository_certified=repository_certified,
    )
    click.echo(str(path))
