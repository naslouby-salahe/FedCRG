from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from pydantic import TypeAdapter

from fedcrg.learning.detectors import DetectorModel
from fedcrg.learning.federated import FederatedServer, equal_client_mean
from fedcrg.types import ClientId

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_CLIENT = _CLIENT_ID_ADAPTER.validate_python("client-a")


class TinyDetector(DetectorModel):
    def __init__(self) -> None:
        super().__init__()
        self.parameter_tensor = torch.nn.Parameter(torch.zeros(2))

    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.zeros((batch.shape[0],))


def _model(values: tuple[float, float]) -> TinyDetector:
    model = TinyDetector()
    with torch.no_grad():
        model.parameter_tensor.copy_(torch.tensor(values))
    return model


def test_equal_mean_aggregation_is_identical_to_mean() -> None:
    target = _model((0.0, 0.0))
    aggregated = equal_client_mean([_model((1.0, 2.0)), _model((3.0, 4.0))])
    target.load_state_dict(aggregated)
    values = target.parameter_tensor.detach().numpy()
    np.testing.assert_allclose(values, [2.0, 3.0])


def test_server_aggregates_client_models() -> None:
    server = FederatedServer(_model((0.0, 0.0)))
    server.aggregate([_model((2.0, 2.0)), _model((4.0, 4.0))])
    aggregated = cast(TinyDetector, server.model)
    values = aggregated.parameter_tensor.detach().numpy()
    np.testing.assert_allclose(values, [3.0, 3.0])


def test_server_returns_state_hash() -> None:
    server = FederatedServer(_model((0.0, 0.0)))
    digest = server.aggregate([_model((2.0, 2.0)), _model((4.0, 4.0))])
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_server_rejects_nan_updates() -> None:
    server = FederatedServer(_model((0.0, 0.0)))
    poisoned = _model((float("nan"), 1.0))
    updates = [_model((1.0, 1.0)), poisoned]
    with pytest.raises((ValueError, FloatingPointError)):
        server.aggregate(updates)


def test_server_rejects_infinite_updates() -> None:
    server = FederatedServer(_model((0.0, 0.0)))
    poisoned = _model((float("inf"), 1.0))
    updates = [_model((1.0, 1.0)), poisoned]
    with pytest.raises((ValueError, FloatingPointError)):
        server.aggregate(updates)


def test_aggregate_rejects_empty_participant_set() -> None:
    server = FederatedServer(_model((0.0, 0.0)))
    with pytest.raises((ValueError, RuntimeError)):
        server.aggregate([])
