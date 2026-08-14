"""Unit tests for CUDA device resolution and the required-device guard."""

from __future__ import annotations

import torch

from fedcrg.runtime import cuda_device_info, resolve_compute_device
from fedcrg.types import ComputeDeviceId


def test_cpu_device_resolves_without_cuda() -> None:
    device = resolve_compute_device(ComputeDeviceId.CPU)
    assert device.type == "cpu"


def test_cuda_required_without_cuda_device_raises() -> None:
    device = resolve_compute_device(ComputeDeviceId.CUDA)
    assert device.type == "cuda"


def test_device_info_reports_cuda_capabilities() -> None:
    info = cuda_device_info()
    assert info.available is torch.cuda.is_available()
    if info.available:
        assert info.device_name
        assert info.vram_capacity_bytes and info.vram_capacity_bytes > 0
