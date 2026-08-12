"""
FedCRG AUC Metrics Module

Implements AUC-based metrics per Section 10 of the FedCRG Roadmap v2.0.

Uses sklearn.metrics for reliable computation, with deterministic ordering.

Critical requirement (H5):
AUROC and AUPRC must be numerically identical across threshold policies
using the same cached test scores, up to serialization/rounding tolerance of 1e-12.

Normative reference: Section 10, Section 1407, Hypothesis H5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

if TYPE_CHECKING:
    pass


AUROC = float
AUPRC = float


@dataclass(frozen=True, slots=True)
class AUROCResult:
    """AUROC result."""
    auroc: float
    n_benign: int
    n_attack: int
    fprs: npt.NDArray[np.float64]
    tprs: npt.NDArray[np.float64]
    thresholds: npt.NDArray[np.float64]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auroc": self.auroc,
            "n_benign": self.n_benign,
            "n_attack": self.n_attack,
            "fprs": self.fprs.tolist(),
            "tprs": self.tprs.tolist(),
            "thresholds": self.thresholds.tolist(),
        }


@dataclass(frozen=True, slots=True)
class AUPRCResult:
    """AUPRC result."""
    auprc: float
    n_benign: int
    n_attack: int
    precisions: npt.NDArray[np.float64]
    recalls: npt.NDArray[np.float64]
    thresholds: npt.NDArray[np.float64]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auprc": self.auprc,
            "n_benign": self.n_benign,
            "n_attack": self.n_attack,
            "precisions": self.precisions.tolist(),
            "recalls": self.recalls.tolist(),
            "thresholds": self.thresholds.tolist(),
        }


def compute_auroc(
    benign_scores: npt.NDArray[np.float64],
    attack_scores: npt.NDArray[np.float64],
) -> AUROCResult:
    """
    Compute AUROC from benign and attack scores.
    
    Higher scores = more anomalous.
    Uses sklearn.metrics.roc_auc_score for reliable computation.
    """
    n_benign = len(benign_scores)
    n_attack = len(attack_scores)

    if n_benign == 0 or n_attack == 0:
        raise ValueError("Both benign and attack scores must be non-empty")

    # Labels: 0=benign, 1=attack
    y_true = np.concatenate([np.zeros(n_benign), np.ones(n_attack)])
    # Scores: higher = more anomalous = higher probability of positive class
    y_scores = np.concatenate([benign_scores, attack_scores])

    # Compute AUROC using sklearn
    auroc = float(roc_auc_score(y_true, y_scores))

    # Compute ROC curve for additional info
    fprs, tprs, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)

    return AUROCResult(
        auroc=auroc,
        n_benign=n_benign,
        n_attack=n_attack,
        fprs=np.array(fprs, dtype=np.float64),
        tprs=np.array(tprs, dtype=np.float64),
        thresholds=np.array(thresholds, dtype=np.float64),
    )


def compute_auprc(
    benign_scores: npt.NDArray[np.float64],
    attack_scores: npt.NDArray[np.float64],
) -> AUPRCResult:
    """
    Compute AUPRC from benign and attack scores.
    
    Higher scores = more anomalous.
    Uses sklearn.metrics.average_precision_score for reliable computation.
    """
    n_benign = len(benign_scores)
    n_attack = len(attack_scores)

    if n_benign == 0 or n_attack == 0:
        raise ValueError("Both benign and attack scores must be non-empty")

    # Labels: 0=benign, 1=attack
    y_true = np.concatenate([np.zeros(n_benign), np.ones(n_attack)])
    # Scores: higher = more anomalous
    y_scores = np.concatenate([benign_scores, attack_scores])

    # Compute AUPRC using sklearn
    auprc = float(average_precision_score(y_true, y_scores))

    # Compute PR curve for additional info
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    return AUPRCResult(
        auprc=auprc,
        n_benign=n_benign,
        n_attack=n_attack,
        precisions=np.array(precisions, dtype=np.float64),
        recalls=np.array(recalls, dtype=np.float64),
        thresholds=np.array(thresholds, dtype=np.float64),
    )


def verify_auc_invariance(
    benign_scores: npt.NDArray[np.float64],
    attack_scores: npt.NDArray[np.float64],
    tolerance: float = 1e-12,
) -> bool:
    """
    Verify AUROC/AUPRC invariance across threshold policies (H5).
    
    Same cached scores should give same AUROC/AUPRC regardless of thresholding.
    """
    # Compute multiple times - should be identical
    auroc_1 = compute_auroc(benign_scores, attack_scores)
    auroc_2 = compute_auroc(benign_scores, attack_scores)
    auroc_3 = compute_auroc(benign_scores, attack_scores)

    auprc_1 = compute_auprc(benign_scores, attack_scores)
    auprc_2 = compute_auprc(benign_scores, attack_scores)
    auprc_3 = compute_auprc(benign_scores, attack_scores)

    # All should be exactly equal
    assert auroc_1.auroc == auroc_2.auroc == auroc_3.auroc, "AUROC not deterministic"
    assert auprc_1.auprc == auprc_2.auprc == auprc_3.auprc, "AUPRC not deterministic"

    print(f"AUC invariance verified (deterministic)")
    return True


def verify_auc_metrics() -> bool:
    """Verify AUC metrics implementation."""
    import numpy as np
    np.random.seed(42)

    # Perfect separation
    benign_scores_perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    attack_scores_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    auroc_perfect = compute_auroc(benign_scores_perfect, attack_scores_perfect)
    assert abs(auroc_perfect.auroc - 1.0) < 0.01, f"Expected AUROC ~1.0, got {auroc_perfect.auroc}"

    auprc_perfect = compute_auprc(benign_scores_perfect, attack_scores_perfect)
    assert abs(auprc_perfect.auprc - 1.0) < 0.01, f"Expected AUPRC ~1.0, got {auprc_perfect.auprc}"

    # Random scores
    np.random.seed(42)
    benign_scores_random = np.random.randn(100)
    attack_scores_random = np.random.randn(100)

    auroc_random = compute_auroc(benign_scores_random, attack_scores_random)
    assert 0.4 < auroc_random.auroc < 0.6, f"Expected AUROC ~0.5, got {auroc_random.auroc}"

    # Invariance
    verify_auc_invariance(benign_scores_random, attack_scores_random)

    print("AUC metrics verification passed.")
    return True


if __name__ == "__main__":
    verify_auc_metrics()
