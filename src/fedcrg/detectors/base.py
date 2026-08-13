"""Detector model contract and deterministic state accounting."""

from __future__ import annotations

import copy
import hashlib
from abc import ABC, abstractmethod

import torch


class DetectorModel(torch.nn.Module, ABC):
    @abstractmethod
    def anomaly_score(self, batch: torch.Tensor) -> torch.Tensor:
        ...

    def clone(self) -> "DetectorModel":
        return copy.deepcopy(self)

    def state_hash(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def trainable_tensor_bytes(self) -> int:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
