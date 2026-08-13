"""Benign-only three-sigma comparator (published FedDetect-style baseline)."""

from __future__ import annotations

import numpy as np

from fedcrg.thresholds.evidence import BenignPolicyEvidence


def three_sigma(clients: tuple[BenignPolicyEvidence, ...]) -> float:
    pooled = np.concatenate(tuple(client.full_policy_budget for client in clients))
    mean = float(np.mean(pooled))
    population_std = float(np.sqrt(np.mean((pooled - mean) ** 2)))
    return mean + 3.0 * population_std
