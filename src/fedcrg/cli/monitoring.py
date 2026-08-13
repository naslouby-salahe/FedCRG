"""``fedcrg monitor``: stream resource telemetry to the console and outputs/monitoring."""

from __future__ import annotations

from pathlib import Path

import click

from fedcrg.runtime.monitoring import ResourceMonitor, write_telemetry


@click.command(name="monitor")
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
