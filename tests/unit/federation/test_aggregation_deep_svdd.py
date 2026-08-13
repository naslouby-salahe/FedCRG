import torch
from fedcrg.configuration.detector_config import AutoencoderConfig, DeepSvddConfig
from fedcrg.detectors.autoencoder import Autoencoder
from fedcrg.detectors.deep_svdd import DeepSvdd
from fedcrg.detectors.create_detector import create_detector
from fedcrg.federation.aggregation import EqualMeanAggregator


def test_equal_mean_aggregation() -> None:
    a = Autoencoder(2, AutoencoderConfig(hidden_dims=(1,), xavier_tanh_gain=5.0 / 3.0))
    b = a.clone()
    with torch.no_grad():
        for param in a.parameters():
            param.fill_(1.0)
        for param in b.parameters():
            param.fill_(3.0)
    target = a.clone()
    EqualMeanAggregator().aggregate_into(target, [a, b])
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
