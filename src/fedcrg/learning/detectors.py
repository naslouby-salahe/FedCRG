"""Autoencoder and Deep-SVDD anomaly detector models."""

from __future__ import annotations

import copy
import hashlib
from abc import ABC, abstractmethod
from typing import cast

import torch
import torch.nn.functional as F

from fedcrg.config import ActivationId, AutoencoderConfig, DeepSvddConfig, DetectorConfig
from fedcrg.types import ByteCount, Dimension, ParameterCount, PositiveCount, Sha256, XavierGain


class DetectorModel(torch.nn.Module, ABC):
    """Base class for anomaly detector models with a common scoring, hashing, and cloning interface."""

    @abstractmethod
    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        """Return one anomaly score per input row; larger means more anomalous."""

    def clone(self) -> DetectorModel:
        """Return a deep copy, independent of this model's parameters and buffers."""
        return copy.deepcopy(self)

    def state_hash(self) -> Sha256:
        """Hash the model's state dict for exact-weight identity checks."""
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def trainable_parameter_count(self) -> PositiveCount:
        """Count parameters that receive gradient updates."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_tensor_bytes(self) -> ByteCount:
        """Total byte size of trainable parameters, used for communication accounting."""
        return sum(p.numel() * p.element_size() for p in self.parameters() if p.requires_grad)


def activation_module(activation: ActivationId) -> type[torch.nn.Module]:
    """Map an activation identifier to the corresponding torch module class."""
    if activation is ActivationId.TANH:
        return torch.nn.Tanh
    raise ValueError(f"Unsupported activation: {activation}")


def build_mlp(
    dims: tuple[Dimension, ...],
    activation: type[torch.nn.Module],
    bias: bool = True,
    output_activation: bool = False,
) -> torch.nn.Sequential:
    """Build a stack of linear layers over `dims`, inserting an activation between each pair."""
    layers: list[torch.nn.Module] = []
    for i, (left, right) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        layers.append(torch.nn.Linear(left, right, bias=bias))
        if output_activation or i < len(dims) - 2:
            layers.append(activation())
    return torch.nn.Sequential(*layers)


def initialize_weights(
    module: torch.nn.Module, gain: XavierGain, forbid_bias: bool = False
) -> None:
    """Apply Xavier-uniform initialization to every linear layer, zeroing biases in place."""
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                if forbid_bias:
                    raise ValueError("Biased linear layers are not allowed in this configuration")
                torch.nn.init.zeros_(m.bias)


def autoencoder_parameter_count(
    input_dim: Dimension, hidden_dims: tuple[Dimension, ...]
) -> ParameterCount:
    """Compute the parameter count of a symmetric autoencoder without instantiating it."""
    encoder = (input_dim, *hidden_dims)
    chain = (*encoder, *tuple(reversed(encoder[:-1])))
    return sum(left * right + right for left, right in zip(chain[:-1], chain[1:], strict=True))


def autoencoder_tensor_bytes(input_dim: Dimension, hidden_dims: tuple[Dimension, ...]) -> ByteCount:
    """Byte size of a symmetric autoencoder's parameters, assuming float32 storage."""
    return autoencoder_parameter_count(input_dim, hidden_dims) * 4


class Autoencoder(DetectorModel):
    """Per-sample anomaly score is mean feature-wise reconstruction MSE."""

    def __init__(self, input_dim: Dimension, config: AutoencoderConfig) -> None:
        super().__init__()
        dims = (input_dim, *config.hidden_dims)
        activation = activation_module(config.activation)

        encoder = build_mlp(dims, activation, output_activation=True)
        decoder = build_mlp(tuple(reversed(dims)), activation, output_activation=False)

        self.network = torch.nn.Sequential(*encoder, *decoder)

        initialize_weights(
            self.network, gain=config.xavier_tanh_gain, forbid_bias=not config.zero_bias
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input batch."""
        return cast(torch.Tensor, self.network(batch))

    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        """Mean feature-wise reconstruction MSE per row."""
        return F.mse_loss(self.forward(batch), batch, reduction="none").mean(dim=1)


class DeepSvdd(DetectorModel):
    """Anomaly score is squared Euclidean distance from a center that is frozen after initialization."""

    center: torch.Tensor

    def __init__(self, input_dim: Dimension, config: DeepSvddConfig) -> None:
        super().__init__()
        activation = activation_module(config.activation)
        dims = (input_dim, *config.hidden_dims, config.embedding_dim)

        self.encoder = build_mlp(dims, activation, bias=config.bias, output_activation=False)
        self.register_buffer("center", torch.zeros(config.embedding_dim))

        gain = torch.nn.init.calculate_gain("tanh")
        initialize_weights(self.encoder, gain=gain)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Embed the input batch."""
        return cast(torch.Tensor, self.encoder(batch))

    def initialize_center(self, batches: list[torch.Tensor]) -> None:
        """Set the center as the equal-client average of per-client mean embeddings; call once, before training."""
        if not batches:
            raise ValueError("At least one batch is required to initialize the center")

        with torch.no_grad():
            client_means = torch.stack(
                [self.forward(batch).mean(dim=0) for batch in batches], dim=0
            )
            self.center.copy_(client_means.mean(dim=0))

    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.sum((self.forward(batch) - self.center) ** 2, dim=1)


def create_detector(input_dim: Dimension, config: DetectorConfig) -> DetectorModel:
    """Instantiate the detector model matching the given config type."""
    if isinstance(config, AutoencoderConfig):
        return Autoencoder(input_dim, config)
    return DeepSvdd(input_dim, config)
