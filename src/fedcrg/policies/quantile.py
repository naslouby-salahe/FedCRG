"""Finite-sample quantile comparators."""

import numpy as np
from fedcrg.policies.base import ClientPolicyData, empirical_quantile

def global_quantile(clients: tuple[ClientPolicyData, ...], alpha: float) -> float:
    return empirical_quantile(np.concatenate(tuple(client.full_benign_policy_scores for client in clients)), alpha)

def local_quantile(client: ClientPolicyData, alpha: float) -> float:
    return empirical_quantile(client.full_benign_policy_scores, alpha)

def three_sigma(clients: tuple[ClientPolicyData, ...]) -> float:
    pooled = np.concatenate(tuple(client.full_benign_policy_scores for client in clients)); mean = float(np.mean(pooled)); population_std = float(np.sqrt(np.mean((pooled - mean) ** 2))); return mean + 3.0 * population_std
