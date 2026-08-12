"""Typed policy inputs and finite-sample quantile convention."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedcrg.protocol.results import ClientProtocolResult


@dataclass(frozen=True, slots=True)
class ClientPolicyData:
    client_id: str
    reference_scores: np.ndarray
    mismatch_scores: np.ndarray
    calibration_scores: np.ndarray
    benign_guard_scores: np.ndarray
    attack_dev_scores: np.ndarray
    benign_test_scores: np.ndarray
    attack_test_scores: np.ndarray
    attack_test_groups: tuple[str, ...]
    protocol: ClientProtocolResult

    @property
    def full_benign_policy_scores(self) -> np.ndarray:
        return np.concatenate((self.reference_scores, self.mismatch_scores, self.calibration_scores))

    @property
    def development_scores(self) -> np.ndarray:
        return np.concatenate((self.benign_guard_scores, self.attack_dev_scores))

    @property
    def development_labels(self) -> np.ndarray:
        return np.concatenate((np.zeros(len(self.benign_guard_scores), dtype=np.int64), np.ones(len(self.attack_dev_scores), dtype=np.int64)))


def empirical_quantile(scores: np.ndarray, alpha: float = 0.01) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float64), kind="stable")
    if len(values) == 0:
        raise ValueError("Cannot compute a threshold from no scores")
    rank = min(len(values), int(np.ceil((len(values) + 1) * (1.0 - alpha))))
    return float(values[rank - 1])
