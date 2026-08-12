"""Deterministic federated client participation."""

import math
import numpy as np

class ClientSampler:
    """Select configured client participation without dependence on input ordering."""
    def __init__(self, client_ids: tuple[str, ...], fraction: float, seed: int) -> None:
        if not client_ids: raise ValueError("At least one client is required")
        if not 0.0 < fraction <= 1.0: raise ValueError("Client fraction must be in (0, 1]")
        self.client_ids = tuple(sorted(client_ids)); self.count = max(1, math.ceil(len(self.client_ids) * fraction)); self.rng = np.random.default_rng(seed)
    def select(self) -> tuple[str, ...]:
        if self.count == len(self.client_ids): return self.client_ids
        values = self.rng.choice(self.client_ids, size=self.count, replace=False).tolist()
        return tuple(sorted(str(value) for value in values))
