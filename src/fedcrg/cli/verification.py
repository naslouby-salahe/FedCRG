"""Artifact verification CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.application.verify import VerifyOutputs


@click.command(name="verify")
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
def verify_command(outputs: Path) -> None:
    runs_root = outputs / "runs"
    if not runs_root.exists():
        click.echo(json.dumps({"valid": True, "runs": 0}, indent=2))
        return
    results: dict[str, object] = {}
    valid = True
    for run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        result = VerifyOutputs().verify_run(run_root)
        results[run_root.name] = {"valid": result.valid, "missing": result.missing, "mismatched": result.mismatched}
        valid = valid and result.valid
    click.echo(json.dumps({"valid": valid, "runs": results}, indent=2))
    if not valid:
        raise click.ClickException("One or more runs failed verification")
