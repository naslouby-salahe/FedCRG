from __future__ import annotations

import torch

from fedcrg.config import AutoencoderConfig, DeepSvddConfig
from fedcrg.learning.detectors import Autoencoder, DeepSvdd, create_detector
from fedcrg.learning.federated import equal_client_mean


def test_equal_mean_aggregation() -> None:
    a = Autoencoder(2, AutoencoderConfig(hidden_dims=(1,), xavier_tanh_gain=5.0 / 3.0))
    b = a.clone()
    with torch.no_grad():
        for param in a.parameters():
            param.fill_(1.0)
        for param in b.parameters():
            param.fill_(3.0)
    target = a.clone()
    target.load_state_dict(equal_client_mean([a, b]))
    for param in target.parameters():
        assert torch.allclose(param, torch.full_like(param, 2.0))


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
