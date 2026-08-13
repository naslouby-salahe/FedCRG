"""CUDA device selection, required-device guard, and device telemetry logging."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from fedcrg.domain.enums import ComputeDeviceId


@dataclass(frozen=True, slots=True)
class CudaDeviceInfo:
    available: bool
    device_name: str | None = None
    vram_capacity_bytes: int | None = None


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
    """Resolve the configured device, refusing silent CPU fallback for CUDA work."""
    if device is ComputeDeviceId.CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "The frozen experiment configuration requires CUDA, but no CUDA device is "
            "available. Refusing to silently fall back to CPU."
        )
    return torch.device(device.value)


def log_device_capabilities(logger: logging.Logger) -> None:
    """Log CUDA availability, selected GPU, GPU name, and VRAM capacity."""
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
    """Log peak allocated VRAM for the current process where possible."""
    if not torch.cuda.is_available():
        return
    index = torch.cuda.current_device()
    logger.info(
        "cuda peak_allocated_vram_bytes=%d",
        int(torch.cuda.max_memory_allocated(index)),
    )
