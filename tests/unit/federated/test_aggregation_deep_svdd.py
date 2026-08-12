import torch
from fedcrg.config.models import AutoencoderConfig, DeepSvddConfig
from fedcrg.detectors.autoencoder import Autoencoder
from fedcrg.detectors.deep_svdd import DeepSvdd
from fedcrg.detectors.factory import DetectorFactory
from fedcrg.federated.aggregation import EqualMeanAggregator


def test_equal_mean_aggregation() -> None:
    a = Autoencoder(2, AutoencoderConfig(hidden_dims=(1,)))
    b = a.clone()
    with torch.no_grad():
        for param in a.parameters(): param.fill_(1.0)
        for param in b.parameters(): param.fill_(3.0)
    target = a.clone()
    EqualMeanAggregator().aggregate_into(target, [a, b])
    for param in target.parameters(): assert torch.allclose(param, torch.full_like(param, 2.0))


def test_deep_svdd_center_and_score() -> None:
    model = DeepSvdd(2, DeepSvddConfig(hidden_dims=(3,), embedding_dim=2))
    batch = torch.randn(4, 2)
    model.initialize_center([batch])
    assert model.anomaly_score(batch).shape == (4,)


def test_detector_factory_builds_both_detectors() -> None:
    factory = DetectorFactory()
    assert isinstance(factory.create(2, AutoencoderConfig(hidden_dims=(1,))), Autoencoder)
    assert isinstance(factory.create(2, DeepSvddConfig(hidden_dims=(2,), embedding_dim=1)), DeepSvdd)
