"""Feed-forward autoencoder anomaly detector."""

from __future__ import annotations

import torch

from fedcrg.config.models import AutoencoderConfig
from fedcrg.detectors.base import DetectorModel


def _activation(name: str) -> type[torch.nn.Module]:
    return torch.nn.Tanh if name == "tanh" else torch.nn.ReLU


class Autoencoder(DetectorModel):
    def __init__(self, input_dim: int, config: AutoencoderConfig) -> None:
        super().__init__()
        dims = (input_dim, *config.hidden_dims)
        activation = _activation(config.activation)
        encoder_layers: list[torch.nn.Module] = []
        for left, right in zip(dims[:-1], dims[1:], strict=True):
            encoder_layers.extend((torch.nn.Linear(left, right), activation()))
        decoder_dims = tuple(reversed(dims))
        decoder_layers: list[torch.nn.Module] = []
        for index, (left, right) in enumerate(zip(decoder_dims[:-1], decoder_dims[1:], strict=True)):
            decoder_layers.append(torch.nn.Linear(left, right))
            if index < len(decoder_dims) - 2:
                decoder_layers.append(activation())
        self.network = torch.nn.Sequential(*encoder_layers, *decoder_layers)
        self._initialize_parameters(config.activation)

    def _initialize_parameters(self, activation: str) -> None:
        gain = torch.nn.init.calculate_gain(activation)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.network(batch)

    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        reconstruction = self.forward(batch)
        return torch.mean((batch - reconstruction) ** 2, dim=1)
