"""Validated equal-client parameter aggregation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from fedcrg.detectors.base import DetectorModel


def equal_client_mean(models: Sequence[DetectorModel]) -> dict[str, torch.Tensor]:
    if not models:
        raise ValueError("At least one client model is required")
    states = [model.state_dict() for model in models]
    keys = tuple(states[0].keys())
    if any(tuple(state.keys()) != keys for state in states[1:]):
        raise ValueError("Client model state dictionaries do not match")
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [state[key].detach() for state in states]
        shape = tensors[0].shape
        dtype = tensors[0].dtype
        if any(tensor.shape != shape for tensor in tensors[1:]):
            raise ValueError(f"Client tensor shapes differ for {key}")
        if any(tensor.dtype != dtype for tensor in tensors[1:]):
            raise ValueError(f"Client tensor dtypes differ for {key}")
        if dtype.is_floating_point:
            if any(not torch.isfinite(tensor).all() for tensor in tensors):
                raise FloatingPointError(f"TRAINING_NUMERICAL_FAILURE: non-finite tensor {key}")
            result[key] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                raise ValueError(f"Non-floating state differs across clients: {key}")
            result[key] = tensors[0].clone()
    return result


class EqualMeanAggregator:
    def aggregate_into(self, target: DetectorModel, clients: Sequence[DetectorModel]) -> None:
        target.load_state_dict(equal_client_mean(clients), strict=True)
