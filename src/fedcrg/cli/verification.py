"""Repository verification CLI."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.experiments.verification import VerifyOutputs


@click.command(name="verify")
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
@click.option("--repository-root", type=click.Path(path_type=Path, exists=True), default=Path("."))
@click.option("--skip-tests", is_flag=True, default=False)
def verify_command(outputs: Path, repository_root: Path, skip_tests: bool) -> None:
    """Fail unless hashes, manifests, workload ledgers, and requested checks reconcile."""
    result = VerifyOutputs().verify_repository(
        outputs,
        run_tests=not skip_tests,
        repository_root=repository_root,
    )
    payload = {
        "valid": result.valid,
        "runs": {
            entry.run_id: {
                "valid": entry.result.valid,
                "missing": entry.result.missing,
                "mismatched": entry.result.mismatched,
            }
            for entry in result.runs
        },
        "experiments": [
            {
                "experiment_id": item.experiment_id.value,
                "complete": item.complete,
                "expected_cells": item.expected_cells,
                "observed_cells": item.observed_cells,
                "problems": item.problems,
            }
            for item in result.experiment_completion
        ],
        "incomplete_experiments": result.incomplete_experiments,
        "test_return_code": result.test_return_code,
    }
    click.echo(json.dumps(payload, indent=2))
    if not result.valid:
        raise click.ClickException("FedCRG verification failed")
