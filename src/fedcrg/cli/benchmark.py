"""Computational-benchmark CLI command."""

from __future__ import annotations

from pathlib import Path

import click

from fedcrg.analysis.computational_benchmark import RunBenchmark
from fedcrg.config.resolve import load_config


@click.command(name="benchmark")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("outputs/reports/latest/benchmark.json"),
)
def benchmark_command(config_path: Path, output: Path) -> None:
    config = load_config(config_path)
    path = RunBenchmark().run_on_synthetic_evidence(config, output)
    click.echo(str(path))
