"""Repository verification CLI."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.application.verify import VerifyOutputs


@click.command(name="verify")
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
@click.option("--repository-root", type=click.Path(path_type=Path, exists=True), default=Path("."))
@click.option("--skip-tests", is_flag=True, default=False)
def verify_command(outputs: Path, repository_root: Path, skip_tests: bool) -> None:
    result = VerifyOutputs().verify_repository(
        outputs,
        run_tests=not skip_tests,
        repository_root=repository_root,
    )
    payload = {
        "valid": result.valid,
        "runs": {
            name: {
                "valid": item.valid,
                "missing": item.missing,
                "mismatched": item.mismatched,
            }
            for name, item in result.runs.items()
        },
        "missing_experiments": result.missing_experiments,
        "test_return_code": result.test_return_code,
    }
    click.echo(json.dumps(payload, indent=2))
    if not result.valid:
        raise click.ClickException("FedCRG verification failed")
