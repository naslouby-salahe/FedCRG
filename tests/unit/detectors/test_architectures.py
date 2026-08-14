"""Unit tests for locked detector architecture contracts."""

from __future__ import annotations

import torch

from fedcrg.config import AutoencoderConfig
from fedcrg.learning.detectors import Autoencoder


def test_locked_autoencoder_parameter_counts_and_zero_biases() -> None:
    nbaiot = Autoencoder(
        115, AutoencoderConfig(hidden_dims=(86, 57, 38, 29), xavier_tanh_gain=5.0 / 3.0)
    )
    diad = Autoencoder(
        86, AutoencoderConfig(hidden_dims=(64, 43, 28, 21), xavier_tanh_gain=5.0 / 3.0)
    )
    assert sum(parameter.numel() for parameter in nbaiot.parameters()) == 36626
    assert sum(parameter.numel() for parameter in diad.parameters()) == 20473
    for model in (nbaiot, diad):
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                assert torch.count_nonzero(module.bias).item() == 0


def test_autoencoder_anomaly_scores_have_reconstruction_shape() -> None:
    model = Autoencoder(4, AutoencoderConfig(hidden_dims=(2, 1), xavier_tanh_gain=5.0 / 3.0))
    scores = model.anomaly_score(torch.randn(6, 4))
    assert scores.shape == (6,)
