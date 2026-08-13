"""Feed-forward autoencoder anomaly detector with protocol-owned initialization."""

from __future__ import annotations

from typing import cast

import torch

from fedcrg.configuration.detector_config import AutoencoderConfig
from fedcrg.domain.enums import ActivationId
from fedcrg.detectors.detector import DetectorModel


def activation_module(activation: ActivationId) -> type[torch.nn.Module]:
    if activation is ActivationId.TANH:
        return torch.nn.Tanh
    if activation is ActivationId.RELU:
        return torch.nn.ReLU
    raise ValueError(f"Unsupported activation: {activation.value}")


def autoencoder_parameter_count(input_dim: int, hidden_dims: tuple[int, ...]) -> int:
    """Derived trainable-parameter count of the symmetric biased autoencoder.

    The full dimension chain is ``input -> hidden -> reversed(hidden) -> input`` and every
    layer is a biased linear map, so the count is ``sum(d_l * d_{l+1} + d_{l+1})`` over
    adjacent layer pairs.
    """
    encoder = (input_dim, *hidden_dims)
    chain = (*encoder, *tuple(reversed(encoder[:-1])))
    return sum(left * right + right for left, right in zip(chain[:-1], chain[1:], strict=True))


def autoencoder_tensor_bytes(input_dim: int, hidden_dims: tuple[int, ...]) -> int:
    """Derived float32 tensor payload of the symmetric autoencoder."""
    return autoencoder_parameter_count(input_dim, hidden_dims) * 4


class Autoencoder(DetectorModel):
    """Symmetric MLP autoencoder whose score is mean feature-wise reconstruction MSE."""

    def __init__(self, input_dim: int, config: AutoencoderConfig) -> None:
        super().__init__()
        dims = (input_dim, *config.hidden_dims)
        activation = activation_module(config.activation)
        encoder_layers: list[torch.nn.Module] = []
        for left, right in zip(dims[:-1], dims[1:], strict=True):
            encoder_layers.extend((torch.nn.Linear(left, right), activation()))

        decoder_dims = tuple(reversed(dims))
        decoder_layers: list[torch.nn.Module] = []
        for index, (left, right) in enumerate(
            zip(decoder_dims[:-1], decoder_dims[1:], strict=True)
        ):
            decoder_layers.append(torch.nn.Linear(left, right))
            if index < len(decoder_dims) - 2:
                decoder_layers.append(activation())

        self.network = torch.nn.Sequential(*encoder_layers, *decoder_layers)
        self._initialize_parameters(config)

    def _initialize_parameters(self, config: AutoencoderConfig) -> None:
        for module in self.modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            torch.nn.init.xavier_uniform_(module.weight, gain=config.xavier_tanh_gain)
            if module.bias is not None:
                if not config.zero_bias:
                    raise ValueError("The frozen autoencoder contract requires zero biases")
                torch.nn.init.zeros_(module.bias)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(batch))

    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        reconstruction = self.forward(batch)
        return torch.mean((batch - reconstruction) ** 2, dim=1)
