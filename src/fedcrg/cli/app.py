"""FedCRG research command-line entry point (``fedcrg``)."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import click
import numpy
import pandas
import scipy
import torch

from fedcrg.cli.analysis_commands import analysis_group, tables_group, verify_command
from fedcrg.cli.data_commands import data_group, environment_group
from fedcrg.cli.experiment_commands import (
    benchmark_command,
    campaign_group,
    evaluate_command,
    experiment_group,
    robustness_group,
    score_command,
    sensitivity_group,
    synthetic_group,
    train_command,
)
from fedcrg.cli.report_commands import report_group, results_group
from fedcrg.configuration.resolve import load_config
from fedcrg.runtime.logging import configure_logging
from fedcrg.runtime.monitoring import ResourceMonitor, write_telemetry


@click.group()
@click.version_option(package_name="fedcrg")
def cli() -> None:
    """FedCRG reproducible research tooling."""
    configure_logging(logs_root=Path("outputs/logs"))


@cli.command(name="doctor")
def doctor() -> None:
    click.echo(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "pandas": pandas.__version__,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            indent=2,
        )
    )


@cli.command(name="monitor")
@click.option(
    "--outputs",
    type=click.Path(path_type=Path),
    default=Path("outputs"),
    show_default=True,
)
@click.option(
    "--interval",
    type=float,
    default=1.0,
    show_default=True,
    help="Seconds between samples.",
)
@click.option(
    "--samples",
    type=int,
    default=None,
    help="Stop after this many samples (default: stream until interrupted).",
)
def monitor_command(outputs: Path, interval: float, samples: int | None) -> None:
    """Stream CPU/RAM/GPU telemetry and persist it under outputs/monitoring/."""
    telemetry_path = outputs / "monitoring" / "telemetry.jsonl"
    monitor = ResourceMonitor()
    if interval <= 0:
        raise click.BadParameter("--interval must be positive")
    click.echo(f"Streaming resource telemetry to {telemetry_path} (Ctrl-C to stop).")
    try:
        for sample in monitor.stream(interval, samples):
            write_telemetry(sample, telemetry_path)
            cuda = sample.cuda
            line = (
                f"ram={sample.process_ram_bytes / 1e6:.1f}MB "
                f"sys_ram_free={sample.available_system_ram_bytes / 1e9:.1f}GB "
                f"cpu={sample.cpu_percent:.1f}%"
            )
            if cuda.available:
                line += (
                    f" gpu={cuda.device_name} vram_used={cuda.allocated_vram_bytes / 1e6:.1f}MB "
                    f"vram_total={cuda.total_vram_bytes / 1e9:.1f}GB"
                )
            else:
                line += " gpu=unavailable"
            click.echo(line)
    except KeyboardInterrupt:
        click.echo("\nStopped.")
        raise SystemExit(0) from None


@cli.group(name="config")
def config_group() -> None:
    """Validate fully resolved experiment configurations."""


@config_group.command(name="validate")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
def validate_config(config_path: Path) -> None:
    config = load_config(config_path)
    click.echo(json.dumps({"valid": True, "config_hash": config.config_hash}, indent=2))


cli.add_command(data_group)
cli.add_command(environment_group)
cli.add_command(tables_group)
cli.add_command(synthetic_group)
cli.add_command(train_command)
cli.add_command(score_command)
cli.add_command(evaluate_command)
cli.add_command(robustness_group)
cli.add_command(sensitivity_group)
cli.add_command(benchmark_command)
cli.add_command(report_group)
cli.add_command(experiment_group)
cli.add_command(verify_command)
cli.add_command(campaign_group)
cli.add_command(results_group)
cli.add_command(analysis_group)


if __name__ == "__main__":
    cli()
