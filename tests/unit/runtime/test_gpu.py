"""Tests for compute device resolution and CUDA info reporting."""

from __future__ import annotations

import pytest
import torch

from fedcrg.runtime import cuda_device_info, resolve_compute_device
from fedcrg.types import ComputeDeviceId


def test_cpu_device_resolves_without_cuda() -> None:
    """Resolving the CPU device never depends on CUDA availability."""
    device = resolve_compute_device(ComputeDeviceId.CPU)
    assert device.type == "cpu"


def test_cuda_required_without_cuda_device_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting CUDA without an available device raises instead of silently falling back."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        resolve_compute_device(ComputeDeviceId.CUDA)


def test_cuda_resolves_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting CUDA when available resolves to a CUDA device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = resolve_compute_device(ComputeDeviceId.CUDA)
    assert device.type == "cuda"


def test_device_info_reports_cuda_capabilities() -> None:
    """CUDA device info reflects real availability and reports VRAM when present."""
    info = cuda_device_info()
    assert info.available is torch.cuda.is_available()
    if info.available:
        assert info.device_name
        assert info.vram_capacity_bytes
        assert info.vram_capacity_bytes > 0
