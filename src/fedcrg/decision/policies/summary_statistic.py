"""Locked supervised-development summary-statistic (published-style) comparator."""

from __future__ import annotations

import numpy as np

from fedcrg.decision.policies.development_f1 import f1_at_threshold
from fedcrg.decision.evidence import SupervisedDevelopmentEvidence
from fedcrg.domain.enums import SupervisedClassLabel
from fedcrg.domain.values import ClassMoments


def mean_client_f1_at_threshold(
    clients: tuple[SupervisedDevelopmentEvidence, ...], threshold: float
) -> float:
    return float(np.mean([f1_at_threshold(client, threshold) for client in clients]))


def _pooled_moments(
    clients: tuple[SupervisedDevelopmentEvidence, ...],
    label: SupervisedClassLabel,
) -> ClassMoments:
    """Pool (count, mean, variance) across clients for one supervised class."""
    counts: list[int] = []
    means: list[float] = []
    variances: list[float] = []
    for client in clients:
        values = client.scores[client.labels == label]
        counts.append(len(values))
        means.append(float(np.mean(values)))
        variances.append(float(np.var(values, ddof=0)))
    total = sum(counts)
    mean = sum(count * local_mean for count, local_mean in zip(counts, means, strict=True)) / total
    variance = (
        sum(
            count * (local_variance + local_mean**2)
            for count, local_mean, local_variance in zip(counts, means, variances, strict=True)
        )
        / total
        - mean**2
    )
    return ClassMoments(mean=mean, std=float(np.sqrt(max(variance, 0.0))))


def summary_statistic_threshold(
    clients: tuple[SupervisedDevelopmentEvidence, ...],
    candidate_count: int,
) -> float | None:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    benign = _pooled_moments(clients, SupervisedClassLabel.BENIGN)
    attack = _pooled_moments(clients, SupervisedClassLabel.ATTACK)

    lower = max(benign.mean - 3.0 * benign.std, attack.mean - 3.0 * attack.std)
    upper = min(benign.mean + 3.0 * benign.std, attack.mean + 3.0 * attack.std)
    if lower >= upper:
        return None

    candidates = np.linspace(lower, upper, candidate_count, dtype=np.float64)
    scores = np.asarray([mean_client_f1_at_threshold(clients, float(t)) for t in candidates])
    best = float(np.max(scores))
    return float(candidates[int(np.flatnonzero(scores == best)[0])])
