"""Deterministic federated client participation."""

from __future__ import annotations

import math

import numpy as np

from fedcrg.core.ids import ClientId


class ClientSampler:
    """Select configured client participation without dependence on input ordering."""

    def __init__(self, client_ids: tuple[ClientId, ...], fraction: float, seed: int) -> None:
        if not client_ids:
            raise ValueError("At least one client is required")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("Client fraction must be in (0, 1]")
        self.client_ids = tuple(sorted(client_ids))
        self.count = max(1, math.ceil(len(self.client_ids) * fraction))
        self.rng = np.random.default_rng(seed)

    def select(self) -> tuple[ClientId, ...]:
        if self.count == len(self.client_ids):
            return self.client_ids
        indices = self.rng.choice(len(self.client_ids), size=self.count, replace=False)
        return tuple(sorted(self.client_ids[int(index)] for index in indices))
