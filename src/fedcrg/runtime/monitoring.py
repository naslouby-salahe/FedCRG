"""Runtime resource telemetry persisted under ``outputs/monitoring/``."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil
import torch

from fedcrg.artifacts.json_io import atomic_write_json


@dataclass(frozen=True, slots=True)
class CudaTelemetry:
    available: bool
    device_count: int = 0
    device_name: str | None = None
    total_vram_bytes: int = 0
    allocated_vram_bytes: int = 0
    reserved_vram_bytes: int = 0
    peak_allocated_vram_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ResourceSample:
    timestamp: str
    process_ram_bytes: int
    available_system_ram_bytes: int
    total_system_ram_bytes: int
    cpu_percent: float
    cuda: CudaTelemetry


class ResourceMonitor:
    """Sample process/system resources and CUDA state on demand."""

    def __init__(self, process: psutil.Process | None = None) -> None:
        self._process = process or psutil.Process()

    def sample(self, timestamp: str | None = None) -> ResourceSample:
        memory = psutil.virtual_memory()
        cuda = self._sample_cuda()
        return ResourceSample(
            timestamp=timestamp or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            process_ram_bytes=self._process.memory_info().rss,
            available_system_ram_bytes=memory.available,
            total_system_ram_bytes=memory.total,
            cpu_percent=self._process.cpu_percent(interval=None),
            cuda=cuda,
        )

    @staticmethod
    def _sample_cuda() -> CudaTelemetry:
        if not torch.cuda.is_available():
            return CudaTelemetry(available=False)
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return CudaTelemetry(
            available=True,
            device_count=torch.cuda.device_count(),
            device_name=torch.cuda.get_device_name(index),
            total_vram_bytes=int(props.total_memory),
            allocated_vram_bytes=int(torch.cuda.memory_allocated(index)),
            reserved_vram_bytes=int(torch.cuda.memory_reserved(index)),
            peak_allocated_vram_bytes=int(torch.cuda.max_memory_allocated(index)),
        )

    def stream(
        self,
        interval_seconds: float,
        max_samples: int | None = None,
    ) -> Iterator[ResourceSample]:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        collected = 0
        while max_samples is None or collected < max_samples:
            yield self.sample()
            collected += 1
            time.sleep(interval_seconds)


def write_telemetry(sample: ResourceSample, output: Path) -> None:
    """Append one sample to ``outputs/monitoring/telemetry.jsonl``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(sample)) + "\n")


def write_telemetry_snapshot(sample: ResourceSample, output: Path) -> None:
    """Atomically persist one JSON snapshot of the latest sample."""
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, asdict(sample))
