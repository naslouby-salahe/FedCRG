"""Tests for resource monitoring, telemetry rendering, and telemetry persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fedcrg.runtime import (
    CudaTelemetry,
    ResourceMonitor,
    ResourceSample,
    render_telemetry,
    write_telemetry,
)


def test_monitor_samples_process_and_system_resources() -> None:
    """A resource sample reports plausible process and system RAM/CPU figures."""
    sample = ResourceMonitor().sample()
    assert sample.process_ram_bytes > 0
    assert sample.total_system_ram_bytes > 0
    assert sample.available_system_ram_bytes <= sample.total_system_ram_bytes
    assert sample.cpu_percent >= 0.0
    assert sample.timestamp


def test_telemetry_appends_jsonl(tmp_path: Path) -> None:
    """Repeated writes append JSONL lines rather than overwriting the file."""
    sample = ResourceSample(
        timestamp="2026-08-13T00:00:00+0000",
        process_ram_bytes=1,
        available_system_ram_bytes=2,
        total_system_ram_bytes=3,
        cpu_percent=4.0,
        cuda=CudaTelemetry(available=False),
    )
    path = tmp_path / "telemetry.jsonl"
    write_telemetry(sample, path)
    write_telemetry(sample, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["cpu_percent"] == 4.0
    assert payload["cuda"]["available"] is False


def test_render_telemetry_is_deterministic_text() -> None:
    """Rendering the same sample twice produces identical text."""
    sample = ResourceSample(
        timestamp="2026-08-13T00:00:00+0000",
        process_ram_bytes=1,
        available_system_ram_bytes=2,
        total_system_ram_bytes=3,
        cpu_percent=4.0,
        cuda=CudaTelemetry(available=False),
    )
    first = render_telemetry(sample)
    second = render_telemetry(sample)
    assert first == second
    assert "cpu" in first


def test_stream_bounds_sample_count() -> None:
    """The stream stops after max_samples rather than running indefinitely."""
    monitor = ResourceMonitor()
    collected = list(monitor.stream(interval_seconds=0.001, max_samples=3))
    assert len(collected) == 3


def test_stream_rejects_nonpositive_interval() -> None:
    """A non-positive sampling interval is rejected rather than causing a busy loop."""
    monitor = ResourceMonitor()
    stream = monitor.stream(interval_seconds=0.0)
    with pytest.raises(ValueError):
        next(stream)
