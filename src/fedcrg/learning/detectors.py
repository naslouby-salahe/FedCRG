"""Anomaly detector models: the federated autoencoder and Deep-SVDD.

Both detectors are deliberately conventional score generators. FedCRG is a
post-training operating-point governance layer; the architectures, optimizer,
and reconstruction scores are not part of the FedCRG contribution.
"""

from __future__ import annotations

import copy
import hashlib
from abc import ABC, abstractmethod
from typing import cast

import torch

from fedcrg.config import ActivationId, AutoencoderConfig, DeepSvddConfig, DetectorConfig
from fedcrg.types import ByteCount, Dimension, ParameterCount, PositiveCount, Sha256


class DetectorModel(torch.nn.Module, ABC):
    @abstractmethod
    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor: ...

    def clone(self) -> DetectorModel:
        return copy.deepcopy(self)

    def state_hash(self) -> Sha256:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def trainable_parameter_count(self) -> PositiveCount:
        return sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )

    def trainable_tensor_bytes(self) -> ByteCount:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def activation_module(activation: ActivationId) -> type[torch.nn.Module]:
    if activation is ActivationId.TANH:
        return torch.nn.Tanh
    raise ValueError(f"Unsupported activation: {activation.value}")


def autoencoder_parameter_count(input_dim: Dimension, hidden_dims: tuple[Dimension, ...]) -> ParameterCount:
    """Derived trainable-parameter count of the symmetric biased autoencoder.

    The full dimension chain is ``input -> hidden -> reversed(hidden) -> input``
    and every layer is a biased linear map, so the count is
    ``sum(d_l * d_{l+1} + d_{l+1})`` over adjacent layer pairs.
    """
    encoder = (input_dim, *hidden_dims)
    chain = (*encoder, *tuple(reversed(encoder[:-1])))
    return sum(left * right + right for left, right in zip(chain[:-1], chain[1:], strict=True))


def autoencoder_tensor_bytes(input_dim: Dimension, hidden_dims: tuple[Dimension, ...]) -> ByteCount:
    """Derived float32 tensor payload of the symmetric autoencoder."""
    return autoencoder_parameter_count(input_dim, hidden_dims) * 4


class Autoencoder(DetectorModel):
    """Symmetric MLP autoencoder whose score is mean feature-wise reconstruction MSE."""

    def __init__(self, input_dim: Dimension, config: AutoencoderConfig) -> None:
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


class DeepSvdd(DetectorModel):
    center: torch.Tensor

    def __init__(self, input_dim: Dimension, config: DeepSvddConfig) -> None:
        super().__init__()
        activation = torch.nn.Tanh if config.activation is ActivationId.TANH else torch.nn.ReLU
        dims = (input_dim, *config.hidden_dims, config.embedding_dim)
        layers: list[torch.nn.Module] = []
        for index, (left, right) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            layers.append(torch.nn.Linear(left, right, bias=config.bias))
            if index < len(dims) - 2:
                layers.append(activation())
        self.encoder = torch.nn.Sequential(*layers)
        self.register_buffer("center", torch.zeros(config.embedding_dim))
        gain = torch.nn.init.calculate_gain("tanh")
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.encoder(batch))

    def initialize_center(self, batches: list[torch.Tensor]) -> None:
        if not batches:
            raise ValueError("At least one batch is required to initialize the center")
        with torch.no_grad():
            client_means = torch.stack(
                [self.forward(batch).mean(dim=0) for batch in batches], dim=0
            )
            self.center.copy_(client_means.mean(dim=0))

    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        embedding = self.forward(batch)
        return torch.sum((embedding - self.center) ** 2, dim=1)


def create_detector(input_dim: Dimension, config: DetectorConfig) -> DetectorModel:
    if isinstance(config, AutoencoderConfig):
        return Autoencoder(input_dim, config)
    if isinstance(config, DeepSvddConfig):
        return DeepSvdd(input_dim, config)
    raise TypeError(f"Unsupported detector configuration: {type(config)!r}")
