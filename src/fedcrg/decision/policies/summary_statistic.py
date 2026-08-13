"""Locked supervised-development summary-statistic (published-style) comparator."""

from __future__ import annotations

import numpy as np

from fedcrg.decision.policies.development_f1 import f1_at_threshold
from fedcrg.decision.evidence import SupervisedDevelopmentEvidence


def mean_client_f1_at_threshold(
    clients: tuple[SupervisedDevelopmentEvidence, ...], threshold: float
) -> float:
    return float(np.mean([f1_at_threshold(client, threshold) for client in clients]))


def summary_statistic_threshold(
    clients: tuple[SupervisedDevelopmentEvidence, ...],
    candidate_count: int,
) -> float | None:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    moments: dict[int, list[tuple[int, float, float]]] = {0: [], 1: []}
    for client in clients:
        for label in (0, 1):
            values = client.scores[client.labels == label]
            moments[label].append(
                (len(values), float(np.mean(values)), float(np.var(values, ddof=0)))
            )

    pooled: dict[int, tuple[float, float]] = {}
    for label, rows in moments.items():
        total = sum(count for count, _, _ in rows)
        mean = sum(count * local_mean for count, local_mean, _ in rows) / total
        variance = (
            sum(
                count * (local_variance + local_mean**2)
                for count, local_mean, local_variance in rows
            )
            / total
            - mean**2
        )
        pooled[label] = (mean, float(np.sqrt(max(variance, 0.0))))

    benign_mean, benign_std = pooled[0]
    attack_mean, attack_std = pooled[1]
    lower = max(benign_mean - 3.0 * benign_std, attack_mean - 3.0 * attack_std)
    upper = min(benign_mean + 3.0 * benign_std, attack_mean + 3.0 * attack_std)
    if lower >= upper:
        return None

    candidates = np.linspace(lower, upper, candidate_count, dtype=np.float64)
    scores = np.asarray([mean_client_f1_at_threshold(clients, float(t)) for t in candidates])
    best = float(np.max(scores))
    return float(candidates[int(np.flatnonzero(scores == best)[0])])
