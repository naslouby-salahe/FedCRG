"""Benign-only pooled-federation quantile comparator."""

from __future__ import annotations

import numpy as np

from fedcrg.thresholds.evidence import BenignPolicyEvidence, empirical_quantile


def global_quantile(clients: tuple[BenignPolicyEvidence, ...], alpha: float) -> float:
    counts = {len(client.full_policy_budget) for client in clients}
    if len(counts) != 1:
        raise ValueError("Global quantile requires equal per-client benign policy budgets")
    pooled = np.concatenate(tuple(client.full_policy_budget for client in clients))
    return empirical_quantile(pooled, alpha)
