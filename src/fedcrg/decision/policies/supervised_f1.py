"""Locked supervised-development global-F1-optimal comparator."""

from __future__ import annotations

import numpy as np

from fedcrg.decision.policies.summary_statistic import mean_client_f1_at_threshold
from fedcrg.decision.evidence import SupervisedDevelopmentEvidence


def supervised_global_f1(
    clients: tuple[SupervisedDevelopmentEvidence, ...],
    candidate_count: int,
) -> float:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    minimum = min(float(np.min(client.scores)) for client in clients)
    maximum = max(float(np.max(client.scores)) for client in clients)
    candidates = np.linspace(minimum, maximum, candidate_count, dtype=np.float64)
    scores = np.asarray([mean_client_f1_at_threshold(clients, float(t)) for t in candidates])
    best = float(np.max(scores))
    return float(candidates[int(np.flatnonzero(scores == best)[0])])
