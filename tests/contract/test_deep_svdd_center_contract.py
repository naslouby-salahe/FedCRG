from __future__ import annotations

import torch

from fedcrg.config import DeepSvddConfig
from fedcrg.learning.detectors import DeepSvdd


def _model() -> DeepSvdd:
    torch.manual_seed(11)
    return DeepSvdd(115, DeepSvddConfig(hidden_dims=(64,), embedding_dim=32))


def test_center_is_equal_mean_of_client_initial_mean_embeddings() -> None:
    model = _model()
    first = torch.zeros((2, 115), dtype=torch.float32)
    second = torch.ones((20, 115), dtype=torch.float32)
    with torch.no_grad():
        expected = torch.stack(
            (
                model(first).mean(dim=0),
                model(second).mean(dim=0),
            ),
            dim=0,
        ).mean(dim=0)
    model.initialize_center([first, second])
    assert torch.equal(model.center, expected)


def test_center_is_not_a_trainable_parameter_and_remains_frozen() -> None:
    model = _model()
    batch = torch.randn((16, 115), generator=torch.Generator().manual_seed(22))
    model.initialize_center([batch])
    frozen = model.center.detach().clone()
    assert "center" not in dict(model.named_parameters())
    assert "center" in dict(model.named_buffers())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss = model.anomaly_score(batch).mean()
    loss.backward()
    optimizer.step()

    assert torch.equal(model.center, frozen)
