from __future__ import annotations

import pytest
import torch

from fedcrg.detectors.base import DetectorModel
from fedcrg.federated.server import FederatedServer


class TinyDetector(DetectorModel):
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(width, width)

    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.linear(batch) ** 2, dim=1)


class RenamedTinyDetector(DetectorModel):
    def __init__(self) -> None:
        super().__init__()
        self.other = torch.nn.Linear(2, 2)

    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.other(batch) ** 2, dim=1)


def test_server_rejects_nonfinite_client_parameters() -> None:
    server = FederatedServer(TinyDetector())
    good = TinyDetector()
    bad = TinyDetector()
    with torch.no_grad():
        next(bad.parameters()).view(-1)[0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        server.aggregate([good, bad])


def test_server_rejects_parameter_name_mismatch() -> None:
    server = FederatedServer(TinyDetector())
    with pytest.raises(ValueError):
        server.aggregate([TinyDetector(), RenamedTinyDetector()])


def test_server_rejects_parameter_shape_mismatch() -> None:
    server = FederatedServer(TinyDetector(width=2))
    with pytest.raises(ValueError):
        server.aggregate([TinyDetector(width=2), TinyDetector(width=3)])


def test_server_rejects_parameter_dtype_mismatch() -> None:
    server = FederatedServer(TinyDetector())
    good = TinyDetector()
    double_precision = TinyDetector().double()
    with pytest.raises(ValueError):
        server.aggregate([good, double_precision])
