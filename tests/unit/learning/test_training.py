from __future__ import annotations

import pytest
import torch
from pydantic import TypeAdapter, ValidationError
from torch.utils.data import TensorDataset

from fedcrg.config import AutoencoderConfig, TrainingConfig
from fedcrg.learning.detectors import Autoencoder
from fedcrg.learning.federated import FederatedTrainer, cosine_learning_rate
from fedcrg.types import (
    AggregationId,
    ClientId,
    ComputeDeviceId,
    OptimizerId,
)

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)


def _clients(count: int) -> dict[ClientId, TensorDataset]:
    return {
        _CLIENT_ID_ADAPTER.validate_python(f"c{i}"): TensorDataset(torch.randn(4, 2))
        for i in range(count)
    }


def _config(fraction: float = 1.0) -> TrainingConfig:
    return TrainingConfig(
        rounds=1,
        local_epochs=1,
        batch_size=2,
        optimizer=OptimizerId.ADAM,
        learning_rate_initial=1e-3,
        learning_rate_final=1e-3,
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        weight_decay=0.0,
        client_fraction=fraction,
        aggregation=AggregationId.EQUAL_CLIENT_MEAN,
        early_stopping=False,
        mixed_precision=False,
        deterministic_algorithms=True,
        record_round20_score_correlation=False,
        device=ComputeDeviceId.CPU,
    )


def test_learning_rate_uses_configured_endpoints() -> None:
    assert cosine_learning_rate(0, 3, 0.2, 0.01) == 0.2
    assert cosine_learning_rate(2, 3, 0.2, 0.01) == 0.01


def test_client_fraction_is_locked_to_full_participation() -> None:
    model = Autoencoder(2, AutoencoderConfig(hidden_dims=(1,), xavier_tanh_gain=5.0 / 3.0))
    _, result = FederatedTrainer().train(model, _clients(4), _config(), model_seed=11)
    assert len(result.rounds[0].selected_clients) == 4
    assert result.rounds[0].round_communication_bytes > 0


def test_client_fraction_below_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _config(fraction=0.5)


def test_cosine_learning_rate_rejects_round_index_at_horizon() -> None:
    with pytest.raises((ValueError, FloatingPointError)):
        cosine_learning_rate(3, 3, 0.2, 0.01)
