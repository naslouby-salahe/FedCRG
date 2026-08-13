import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import TensorDataset
from fedcrg.config.training_config import AutoencoderConfig, TrainingConfig
from fedcrg.domain.identifiers import ClientId
from fedcrg.detectors.autoencoder import Autoencoder
from fedcrg.federation.learning_rate import cosine_learning_rate
from fedcrg.federation.training import FederatedTrainer


def test_learning_rate_uses_configured_endpoints() -> None:
    assert cosine_learning_rate(0, 3, 0.2, 0.01) == 0.2
    assert cosine_learning_rate(2, 3, 0.2, 0.01) == 0.01


def test_client_fraction_is_locked_to_full_participation() -> None:
    model = Autoencoder(2, AutoencoderConfig(hidden_dims=(1,)))
    datasets = {ClientId(f"c{i}"): TensorDataset(torch.randn(4, 2)) for i in range(4)}
    config = TrainingConfig(
        rounds=1, local_epochs=1, batch_size=2, learning_rate_initial=1e-3, learning_rate_final=1e-3
    )
    _, result = FederatedTrainer().train(model, datasets, config, model_seed=11)
    assert len(result.rounds[0].selected_clients) == 4


def test_client_fraction_below_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(
            rounds=1,
            local_epochs=1,
            batch_size=2,
            client_fraction=0.5,
            learning_rate_initial=1e-3,
            learning_rate_final=1e-3,
        )
