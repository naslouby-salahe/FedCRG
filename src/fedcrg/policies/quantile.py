"""Benign-only shared, local, and published-style threshold comparators."""

from __future__ import annotations

import numpy as np

from fedcrg.policies.base import BenignPolicyEvidence, empirical_quantile


def global_quantile(clients: tuple[BenignPolicyEvidence, ...], alpha: float) -> float:
    counts = {len(client.full_policy_budget) for client in clients}
    if len(counts) != 1:
        raise ValueError("Global quantile requires equal per-client benign policy budgets")
    pooled = np.concatenate(tuple(client.full_policy_budget for client in clients))
    return empirical_quantile(pooled, alpha)


def local_quantile(client: BenignPolicyEvidence, alpha: float) -> float:
    return empirical_quantile(client.full_policy_budget, alpha)


def three_sigma(clients: tuple[BenignPolicyEvidence, ...]) -> float:
    pooled = np.concatenate(tuple(client.full_policy_budget for client in clients))
    mean = float(np.mean(pooled))
    population_std = float(np.sqrt(np.mean((pooled - mean) ** 2)))
    return mean + 3.0 * population_std
