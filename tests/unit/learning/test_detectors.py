"""Tests for the autoencoder and Deep-SVDD detector models."""

from __future__ import annotations

import torch

from fedcrg.config import AutoencoderConfig, DeepSvddConfig
from fedcrg.learning.detectors import Autoencoder, DeepSvdd, create_detector


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


def test_deep_svdd_center_and_score() -> None:
    model = DeepSvdd(2, DeepSvddConfig(hidden_dims=(3,), embedding_dim=2))
    batch = torch.randn(4, 2)
    model.initialize_center([batch])
    assert model.anomaly_score(batch).shape == (4,)


def test_create_detector_builds_both_detectors() -> None:
    assert isinstance(
        create_detector(2, AutoencoderConfig(hidden_dims=(1,), xavier_tanh_gain=5.0 / 3.0)),
        Autoencoder,
    )
    assert isinstance(
        create_detector(2, DeepSvddConfig(hidden_dims=(2,), embedding_dim=1)), DeepSvdd
    )
