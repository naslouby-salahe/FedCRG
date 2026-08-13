import torch

from fedcrg.config.models import AutoencoderConfig
from fedcrg.detectors.autoencoder import Autoencoder


def test_locked_autoencoder_parameter_counts_and_zero_biases() -> None:
    nbaiot = Autoencoder(115, AutoencoderConfig(hidden_dims=(86, 57, 38, 29)))
    diad = Autoencoder(86, AutoencoderConfig(hidden_dims=(64, 43, 28, 21)))
    assert sum(parameter.numel() for parameter in nbaiot.parameters()) == 36626
    assert sum(parameter.numel() for parameter in diad.parameters()) == 20473
    for model in (nbaiot, diad):
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                assert torch.count_nonzero(module.bias).item() == 0
