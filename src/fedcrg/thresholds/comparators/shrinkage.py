"""Threshold-space partial-pooling comparator with the locked tuning rule."""

from __future__ import annotations

import numpy as np

from fedcrg.domain.constants import SHRINKAGE_N0_CANDIDATES
from fedcrg.thresholds.evidence import BenignPolicyEvidence, empirical_quantile


def _estimated_fpr(scores: np.ndarray, threshold: float) -> float:
    if len(scores) == 0:
        raise ValueError("Mismatch evidence cannot be empty when tuning shrinkage")
    return float(np.mean(np.asarray(scores, dtype=np.float64) > threshold))


def tune_shrinkage(clients: tuple[BenignPolicyEvidence, ...], alpha: float) -> int:
    best_n0 = SHRINKAGE_N0_CANDIDATES[0]
    best_error = float("inf")
    for n0 in SHRINKAGE_N0_CANDIDATES:
        errors: list[float] = []
        for client in clients:
            local = empirical_quantile(client.calibration_scores, alpha)
            n_calibration = len(client.calibration_scores)
            weight = n_calibration / (n_calibration + n0)
            threshold = weight * local + (1.0 - weight) * client.evaluation.reference.value
            errors.append(abs(_estimated_fpr(client.mismatch_scores, threshold) - alpha))
        mean_error = float(np.mean(errors))
        if mean_error < best_error or (
            np.isclose(mean_error, best_error, rtol=0.0, atol=1e-15) and n0 > best_n0
        ):
            best_error = mean_error
            best_n0 = n0
    return best_n0


def shrinkage(client: BenignPolicyEvidence, alpha: float, n0: int) -> float:
    local = empirical_quantile(client.calibration_scores, alpha)
    n_calibration = len(client.calibration_scores)
    weight = n_calibration / (n_calibration + n0)
    return weight * local + (1.0 - weight) * client.evaluation.reference.value
