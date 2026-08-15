from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import psutil
import torch
from rich.console import Console
from rich.table import Table

from fedcrg.types import (
    ByteCount,
    CampaignId,
    ComputeDeviceId,
    Duration,
    Identifier,
    LogLevel,
    NonNegativeCount,
    Percentage,
    PositiveCount,
    SampleCount,
    Timestamp,
)

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_FILENAME = "fedcrg.log"


class CampaignStatusColumn(StrEnum):
    STATUS = "status"
    COMPLETED = "completed"
    TOTAL = "total"
    CURRENT = "current"
    ELAPSED_SECONDS = "elapsed_s"


class CacheOutcome(StrEnum):
    HIT = "hit"
    MISS = "miss"


def configure_logging(logs_root: Path | None = None, level: LogLevel | None = None) -> None:
    resolved_level = (level or os.environ.get("FEDCRG_LOG_LEVEL", "INFO")).upper()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if logs_root is not None:
        logs_root.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_root / _LOG_FILENAME, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        handlers.append(file_handler)
    logging.basicConfig(
        level=resolved_level,
        format=_FORMAT,
        datefmt="%H:%M:%S",
        handlers=handlers,
        # force=True so a later call (e.g. a new run reconfiguring the log path)
        # replaces prior handlers instead of stacking duplicate output.
        force=True,
    )


def get_logger(name: Identifier) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def log_stage(logger: logging.Logger, message: Identifier, **fields: object) -> Generator[None]:
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("start %s %s", message, suffix)
    started = time.monotonic()
    try:
        yield
    except Exception:
        logger.exception("failed %s after %.1fs", message, time.monotonic() - started)
        raise
    else:
        logger.info("done %s in %.1fs", message, time.monotonic() - started)


@dataclass(frozen=True, slots=True)
class CudaDeviceInfo:
    available: bool
    device_name: Identifier | None = None
    vram_capacity_bytes: ByteCount | None = None


def cuda_device_info() -> CudaDeviceInfo:
    if not torch.cuda.is_available():
        return CudaDeviceInfo(available=False)
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return CudaDeviceInfo(
        available=True,
        device_name=torch.cuda.get_device_name(index),
        vram_capacity_bytes=int(props.total_memory),
    )


def resolve_compute_device(device: ComputeDeviceId) -> torch.device:
    """Raises rather than silently substituting CPU when CUDA is requested but absent."""
    if device is ComputeDeviceId.CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "The frozen experiment configuration requires CUDA, but no CUDA device is "
            "available. Refusing to silently fall back to CPU."
        )
    return torch.device(device)


def log_device_capabilities(logger: logging.Logger) -> None:
    info = cuda_device_info()
    if not info.available:
        logger.info("cuda unavailable device=cpu")
        return
    logger.info(
        "cuda available device=%s vram_capacity_bytes=%d",
        info.device_name,
        info.vram_capacity_bytes,
    )


def log_peak_vram(logger: logging.Logger) -> None:
    if not torch.cuda.is_available():
        return
    index = torch.cuda.current_device()
    logger.info(
        "cuda peak_allocated_vram_bytes=%d",
        int(torch.cuda.max_memory_allocated(index)),
    )


@dataclass(frozen=True, slots=True)
class CudaTelemetry:
    available: bool
    device_count: NonNegativeCount = 0
    device_name: Identifier | None = None
    total_vram_bytes: ByteCount = 0
    allocated_vram_bytes: ByteCount = 0
    reserved_vram_bytes: ByteCount = 0
    peak_allocated_vram_bytes: ByteCount = 0


@dataclass(frozen=True, slots=True)
class ResourceSample:
    timestamp: Timestamp
    process_ram_bytes: ByteCount
    available_system_ram_bytes: ByteCount
    total_system_ram_bytes: ByteCount
    cpu_percent: Percentage
    cuda: CudaTelemetry


class ResourceMonitor:
    def __init__(self, process: psutil.Process | None = None) -> None:
        self._process = process or psutil.Process()

    def sample(self, timestamp: Timestamp | None = None) -> ResourceSample:
        memory = psutil.virtual_memory()
        cuda = self._sample_cuda()
        return ResourceSample(
            timestamp=timestamp or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            process_ram_bytes=self._process.memory_info().rss,
            available_system_ram_bytes=memory.available,
            total_system_ram_bytes=memory.total,
            # interval=None is non-blocking but measures against the previous call;
            # the first sample from a given process is therefore always 0.0.
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
        interval_seconds: Duration,
        max_samples: SampleCount | None = None,
    ) -> Iterator[ResourceSample]:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        collected = 0
        while max_samples is None or collected < max_samples:
            yield self.sample()
            collected += 1
            time.sleep(interval_seconds)


def write_telemetry(sample: ResourceSample, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(sample)) + "\n")


def _telemetry_text(sample: ResourceSample) -> Identifier:
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
    return line


def render_telemetry(sample: ResourceSample) -> Identifier:
    return _telemetry_text(sample)


def render_campaign_status(
    campaign_id: CampaignId,
    status: Identifier,
    completed: NonNegativeCount,
    total: PositiveCount,
    current_experiment: Identifier | None,
    elapsed_seconds: Duration,
) -> None:
    console = Console()
    table = Table(title=f"campaign {campaign_id}")
    table.add_column(CampaignStatusColumn.STATUS)
    table.add_column(CampaignStatusColumn.COMPLETED)
    table.add_column(CampaignStatusColumn.TOTAL)
    table.add_column(CampaignStatusColumn.CURRENT)
    table.add_column(CampaignStatusColumn.ELAPSED_SECONDS)
    table.add_row(
        status,
        str(completed),
        str(total),
        current_experiment or "-",
        f"{elapsed_seconds:.1f}",
    )
    console.print(table)


def render_cache_status(
    cache_kind: Identifier,
    hit: bool,
    target: Identifier,
    detail: Identifier | None = None,
) -> None:
    console = Console()
    outcome = CacheOutcome.HIT if hit else CacheOutcome.MISS
    suffix = f" ({detail})" if detail else ""
    # markup=False: the literal "[hit]"/"[miss]" prefix would otherwise be parsed
    # as a Rich style tag.
    console.print(f"[{outcome}] {cache_kind} {target}{suffix}", markup=False)
