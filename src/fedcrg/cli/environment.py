"""Environment lock-file freezing CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from fedcrg.artifacts.environment_lock import EnvironmentLocker


@click.group(name="environment")
def environment_group() -> None:
    """Freeze the validated Python environment before confirmatory runs."""


@environment_group.command(name="freeze")
@click.option("--output", type=click.Path(path_type=Path), default=Path("requirements.lock"))
def freeze_environment(output: Path) -> None:
    lock = EnvironmentLocker().freeze(output)
    click.echo(f"lock={lock.path}\nsha256={lock.sha256.value}")
