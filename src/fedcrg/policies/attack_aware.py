"""Locked supervised-development comparators.

These functions require an explicit supervised evidence view so attack development
labels cannot accidentally enter benign-only policy code.
"""

from __future__ import annotations

import numpy as np

from fedcrg.core.constants import SUPERVISED_THRESHOLD_CANDIDATES
from fedcrg.metrics.classification import confusion_matrix, f1
from fedcrg.policies.base import SupervisedDevelopmentEvidence


def _defined_f1(client: SupervisedDevelopmentEvidence, threshold: float) -> float:
    value = f1(confusion_matrix(client.scores, client.labels, threshold))
    return -1.0 if value is None else value


def _mean_client_f1(
    clients: tuple[SupervisedDevelopmentEvidence, ...], threshold: float
) -> float:
    return float(np.mean([_defined_f1(client, threshold) for client in clients]))


def dev_local_global(
    client: SupervisedDevelopmentEvidence,
    global_threshold: float,
    local_threshold: float,
) -> float:
    global_score = _defined_f1(client, global_threshold)
    local_score = _defined_f1(client, local_threshold)
    return local_threshold if local_score > global_score else global_threshold


def summary_statistic_threshold(
    clients: tuple[SupervisedDevelopmentEvidence, ...],
) -> float | None:
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

    candidates = np.linspace(
        lower, upper, SUPERVISED_THRESHOLD_CANDIDATES, dtype=np.float64
    )
    scores = np.asarray([_mean_client_f1(clients, float(t)) for t in candidates])
    best = float(np.max(scores))
    return float(candidates[int(np.flatnonzero(scores == best)[0])])


def supervised_global_f1(
    clients: tuple[SupervisedDevelopmentEvidence, ...],
) -> float:
    minimum = min(float(np.min(client.scores)) for client in clients)
    maximum = max(float(np.max(client.scores)) for client in clients)
    candidates = np.linspace(
        minimum, maximum, SUPERVISED_THRESHOLD_CANDIDATES, dtype=np.float64
    )
    scores = np.asarray([_mean_client_f1(clients, float(t)) for t in candidates])
    best = float(np.max(scores))
    return float(candidates[int(np.flatnonzero(scores == best)[0])])
