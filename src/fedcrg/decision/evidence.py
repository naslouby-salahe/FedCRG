"""Typed threshold-comparator evidence views and the shared finite-sample quantile rule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedcrg.domain.identifiers import AttackGroupId, ClientId
from fedcrg.decision.results import ClientEvaluationResult


@dataclass(frozen=True, slots=True)
class BenignPolicyEvidence:
    client_id: ClientId
    reference_scores: np.ndarray
    mismatch_scores: np.ndarray
    calibration_scores: np.ndarray
    evaluation: ClientEvaluationResult

    def __post_init__(self) -> None:
        for name in ("reference_scores", "mismatch_scores", "calibration_scores"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
                raise ValueError(f"{name} must be a finite non-empty vector")
            object.__setattr__(self, name, values)

    @property
    def full_policy_budget(self) -> np.ndarray:
        return np.concatenate(
            (self.reference_scores, self.mismatch_scores, self.calibration_scores)
        )


@dataclass(frozen=True, slots=True)
class SupervisedDevelopmentEvidence:
    benign: BenignPolicyEvidence
    benign_guard_scores: np.ndarray
    attack_dev_scores: np.ndarray

    def __post_init__(self) -> None:
        guard = np.asarray(self.benign_guard_scores, dtype=np.float64)
        attack = np.asarray(self.attack_dev_scores, dtype=np.float64)
        if len(guard) != 500 or len(attack) != 500:
            raise ValueError(
                "Supervised development must be exactly 500 benign + 500 malicious scores"
            )
        if not np.isfinite(guard).all() or not np.isfinite(attack).all():
            raise ValueError("Supervised development scores must be finite")
        object.__setattr__(self, "benign_guard_scores", guard)
        object.__setattr__(self, "attack_dev_scores", attack)

    @property
    def client_id(self) -> ClientId:
        return self.benign.client_id

    @property
    def scores(self) -> np.ndarray:
        return np.concatenate((self.benign_guard_scores, self.attack_dev_scores))

    @property
    def labels(self) -> np.ndarray:
        return np.concatenate(
            (
                np.zeros(len(self.benign_guard_scores), dtype=np.int64),
                np.ones(len(self.attack_dev_scores), dtype=np.int64),
            )
        )


@dataclass(frozen=True, slots=True)
class FinalTestEvidence:
    """Final-label evidence opened only after non-oracle thresholds are frozen."""

    benign: BenignPolicyEvidence
    benign_test_scores: np.ndarray
    attack_test_scores: np.ndarray
    attack_test_groups: tuple[AttackGroupId, ...]

    def __post_init__(self) -> None:
        benign = np.asarray(self.benign_test_scores, dtype=np.float64)
        attack = np.asarray(self.attack_test_scores, dtype=np.float64)
        if len(benign) == 0 or len(attack) == 0:
            raise ValueError("Final evaluation requires benign and malicious evidence")
        if len(attack) != len(self.attack_test_groups):
            raise ValueError("Attack-test groups must align with attack-test scores")
        if not np.isfinite(benign).all() or not np.isfinite(attack).all():
            raise ValueError("Final test scores must be finite")
        object.__setattr__(self, "benign_test_scores", benign)
        object.__setattr__(self, "attack_test_scores", attack)


def empirical_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float64), kind="stable")
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Quantile thresholds require a non-empty one-dimensional array")
    rank = min(
        len(values),
        int(np.ceil((len(values) + 1) * (1.0 - alpha))),
    )
    return float(values[rank - 1])
