"""Locked supervised-development global-F1-optimal comparator."""

from __future__ import annotations

import numpy as np

from fedcrg.domain.constants import SUPERVISED_THRESHOLD_CANDIDATES
from fedcrg.thresholds.comparators.summary_statistic import mean_client_f1_at_threshold
from fedcrg.thresholds.evidence import SupervisedDevelopmentEvidence


def supervised_global_f1(
    clients: tuple[SupervisedDevelopmentEvidence, ...],
) -> float:
    minimum = min(float(np.min(client.scores)) for client in clients)
    maximum = max(float(np.max(client.scores)) for client in clients)
    candidates = np.linspace(minimum, maximum, SUPERVISED_THRESHOLD_CANDIDATES, dtype=np.float64)
    scores = np.asarray([mean_client_f1_at_threshold(clients, float(t)) for t in candidates])
    best = float(np.max(scores))
    return float(candidates[int(np.flatnonzero(scores == best)[0])])
