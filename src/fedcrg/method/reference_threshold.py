"""Federation reference-threshold estimation."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from fedcrg.domain.identifiers import ClientId
from fedcrg.method.results import ReferenceThreshold


def reference_rank(sample_count: int, alpha: float) -> int:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return min(sample_count, math.ceil((sample_count + 1) * (1.0 - alpha)))


class ReferenceThresholdEstimator:
    def estimate(
        self, scores_by_client: Mapping[ClientId, np.ndarray], alpha: float
    ) -> ReferenceThreshold:
        if not scores_by_client:
            raise ValueError("At least one client must contribute reference scores")
        lengths = {len(np.asarray(scores)) for scores in scores_by_client.values()}
        if 0 in lengths:
            raise ValueError("Reference score arrays must be non-empty")
        if len(lengths) != 1:
            raise ValueError("Each client must contribute the same number of reference scores")
        samples_per_client = next(iter(lengths))
        pooled = np.concatenate(
            [np.asarray(scores, dtype=np.float64) for scores in scores_by_client.values()]
        )
        rank = reference_rank(len(pooled), alpha)
        threshold = float(np.sort(pooled, kind="stable")[rank - 1])
        return ReferenceThreshold(
            value=threshold,
            rank=rank,
            sample_count=len(pooled),
            client_count=len(scores_by_client),
            samples_per_client=samples_per_client,
        )
