"""Deep-SVDD style one-class detector."""

from __future__ import annotations

import torch

from fedcrg.config.models import DeepSvddConfig
from fedcrg.core.enums import ActivationId
from fedcrg.detectors.base import DetectorModel


class DeepSvdd(DetectorModel):
    def __init__(self, input_dim: int, config: DeepSvddConfig) -> None:
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
        gain = torch.nn.init.calculate_gain(config.activation.value)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.encoder(batch)

    def initialize_center(self, batches: list[torch.Tensor]) -> None:
        if not batches:
            raise ValueError("At least one batch is required to initialize the center")
        with torch.no_grad():
            client_means = torch.stack([self.forward(batch).mean(dim=0) for batch in batches], dim=0)
            self.center.copy_(client_means.mean(dim=0))

    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        embedding = self.forward(batch)
        return torch.sum((embedding - self.center) ** 2, dim=1)
