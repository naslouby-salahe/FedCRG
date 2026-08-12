"""
FedCRG Classification Metrics Module

Implements standard classification metrics per Section 10 of the FedCRG Roadmap v2.0.

Metrics:
- FPR (False Positive Rate)
- TPR (True Positive Rate)
- Precision
- Recall
- F1 Score
- Confusion Matrix

Normative reference: Section 10 (Metric registry)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    pass


# =============================================================================
# TYPE ALIASES
# =============================================================================

# For type hints
FPR = float
TPR = float
Precision = float
Recall = float
F1Score = float


# =============================================================================
# CONFUSION MATRIX
# =============================================================================


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """
    Confusion matrix for binary classification.

    Attributes:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives
    """

    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def total(self) -> int:
        """Total number of samples."""
        return self.tp + self.tn + self.fp + self.fn

    @property
    def positive_predictions(self) -> int:
        """Total positive predictions (TP + FP)."""
        return self.tp + self.fp

    @property
    def negative_predictions(self) -> int:
        """Total negative predictions (TN + FN)."""
        return self.tn + self.fn

    @property
    def actual_positives(self) -> int:
        """Total actual positives (TP + FN)."""
        return self.tp + self.fn

    @property
    def actual_negatives(self) -> int:
        """Total actual negatives (TN + FP)."""
        return self.tn + self.fp

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "total": self.total,
        }

    @classmethod
    def from_arrays(
        cls,
        y_true: npt.NDArray[np.int64],
        y_pred: npt.NDArray[np.int64],
    ) -> "ConfusionMatrix":
        """
        Create confusion matrix from true and predicted labels.

        Args:
            y_true: True labels (0=negative, 1=positive)
            y_pred: Predicted labels (0=negative, 1=positive)

        Returns:
            ConfusionMatrix
        """
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        return cls(tp=tp, tn=tn, fp=fp, fn=fn)


# =============================================================================
# COMPUTATION FUNCTIONS
# =============================================================================


def compute_confusion_matrix(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
) -> ConfusionMatrix:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels (0=negative, 1=positive)
        y_pred: Predicted labels (0=negative, 1=positive)

    Returns:
        ConfusionMatrix
    """
    return ConfusionMatrix.from_arrays(y_true, y_pred)


def compute_fpr(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
) -> FPR:
    """
    Compute False Positive Rate.

    FPR = FP / (FP + TN) = FP / actual_negatives

    Args:
        y_true: True labels (0=negative, 1=positive)
        y_pred: Predicted labels (0=negative, 1=positive)

    Returns:
        False Positive Rate (float)
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    actual_negatives = cm.actual_negatives

    if actual_negatives == 0:
        # No negative samples - FPR is undefined
        return float('nan')

    return cm.fp / actual_negatives


def compute_tpr(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
) -> TPR:
    """
    Compute True Positive Rate (Recall/Sensitivity).

    TPR = TP / (TP + FN) = TP / actual_positives

    Args:
        y_true: True labels (0=negative, 1=positive)
        y_pred: Predicted labels (0=negative, 1=positive)

    Returns:
        True Positive Rate (float)
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    actual_positives = cm.actual_positives

    if actual_positives == 0:
        # No positive samples - TPR is undefined
        return float('nan')

    return cm.tp / actual_positives


def compute_precision(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
) -> Precision:
    """
    Compute Precision.

    Precision = TP / (TP + FP)

    Args:
        y_true: True labels (0=negative, 1=positive)
        y_pred: Predicted labels (0=negative, 1=positive)

    Returns:
        Precision (float)
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    positive_predictions = cm.positive_predictions

    if positive_predictions == 0:
        # No positive predictions - precision is undefined
        return float('nan')

    return cm.tp / positive_predictions


def compute_recall(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
) -> Recall:
    """
    Compute Recall (same as TPR).

    Recall = TP / (TP + FN)

    Args:
        y_true: True labels (0=negative, 1=positive)
        y_pred: Predicted labels (0=negative, 1=positive)

    Returns:
        Recall (float)
    """
    return compute_tpr(y_true, y_pred)


def compute_f1(
    y_true: npt.NDArray[np.int64],
    y_pred: npt.NDArray[np.int64],
) -> F1Score:
    """
    Compute F1 Score.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        y_true: True labels (0=negative, 1=positive)
        y_pred: Predicted labels (0=negative, 1=positive)

    Returns:
        F1 Score (float)
    """
    precision = compute_precision(y_true, y_pred)
    recall = compute_recall(y_true, y_pred)

    if precision + recall == 0:
        # Both precision and recall are 0 or undefined - F1 is undefined
        return float('nan')

    return 2.0 * (precision * recall) / (precision + recall)


# =============================================================================
# BATCH COMPUTATION
# =============================================================================


def compute_fpr_from_counts(
    fp: int,
    tn: int,
) -> FPR:
    """
    Compute FPR from FP and TN counts.

    Args:
        fp: Number of false positives
        tn: Number of true negatives

    Returns:
        False Positive Rate (float)
    """
    actual_negatives = tn + fp
    if actual_negatives == 0:
        return float('nan')
    return fp / actual_negatives


def compute_tpr_from_counts(
    tp: int,
    fn: int,
) -> TPR:
    """
    Compute TPR from TP and FN counts.

    Args:
        tp: Number of true positives
        fn: Number of false negatives

    Returns:
        True Positive Rate (float)
    """
    actual_positives = tp + fn
    if actual_positives == 0:
        return float('nan')
    return tp / actual_positives


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================


def verify_classification_metrics() -> bool:
    """
    Verify classification metrics implementation.

    Tests:
    - Confusion matrix computation
    - FPR, TPR, precision, recall, F1 computation
    - Edge cases (no positives, no negatives, no predictions)
    """
    # Test data: 100 samples
    # 60 negative (0), 40 positive (1)
    # Predict: 70 negative (0), 30 positive (1)
    # TP: 35, TN: 55, FP: 5, FN: 5
    y_true = np.array([0] * 60 + [1] * 40)
    y_pred = np.array([0] * 55 + [1] * 5 + [0] * 5 + [1] * 35)

    # Verify confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.tp == 35, f"Expected TP=35, got {cm.tp}"
    assert cm.tn == 55, f"Expected TN=55, got {cm.tn}"
    assert cm.fp == 5, f"Expected FP=5, got {cm.fp}"
    assert cm.fn == 5, f"Expected FN=5, got {cm.fn}"
    assert cm.total == 100
    assert cm.actual_positives == 40
    assert cm.actual_negatives == 60
    assert cm.positive_predictions == 40
    assert cm.negative_predictions == 60

    # Verify FPR = FP / (FP + TN) = 5 / 60 = 0.083333...
    # actual_negatives = 60 (all samples with y_true=0)
    expected_fpr = 5.0 / 60.0
    computed_fpr = compute_fpr(y_true, y_pred)
    assert abs(computed_fpr - expected_fpr) < 1e-10, f"Expected FPR={expected_fpr}, got {computed_fpr}"

    # Verify TPR = TP / (TP + FN) = 35 / 40 = 0.875
    expected_tpr = 35.0 / 40.0
    computed_tpr = compute_tpr(y_true, y_pred)
    assert abs(computed_tpr - expected_tpr) < 1e-10, f"Expected TPR={expected_tpr}, got {computed_tpr}"

    # Verify precision = TP / (TP + FP) = 35 / 40 = 0.875
    expected_precision = 35.0 / 40.0
    computed_precision = compute_precision(y_true, y_pred)
    assert abs(computed_precision - expected_precision) < 1e-10, f"Expected precision={expected_precision}, got {computed_precision}"

    # Verify recall = TPR = 35 / 40 = 0.875
    computed_recall = compute_recall(y_true, y_pred)
    assert abs(computed_recall - expected_tpr) < 1e-10

    # Verify F1 = 2 * (precision * recall) / (precision + recall) = 2 * 0.875 * 0.875 / 1.75
    expected_f1 = 2.0 * (0.875 * 0.875) / (0.875 + 0.875)
    computed_f1 = compute_f1(y_true, y_pred)
    assert abs(computed_f1 - expected_f1) < 1e-10, f"Expected F1={expected_f1}, got {computed_f1}"

    # Test from counts
    # FPR from counts: fp=5, tn=55 => actual_negatives = 5+55 = 60
    assert abs(compute_fpr_from_counts(5, 55) - expected_fpr) < 1e-10
    assert abs(compute_tpr_from_counts(35, 5) - expected_tpr) < 1e-10

    # Test edge cases
    # No positive samples
    y_true_no_pos = np.array([0] * 10)
    y_pred_no_pos = np.array([0] * 5 + [1] * 5)
    assert np.isnan(compute_tpr(y_true_no_pos, y_pred_no_pos))
    assert np.isnan(compute_recall(y_true_no_pos, y_pred_no_pos))
    assert np.isnan(compute_f1(y_true_no_pos, y_pred_no_pos))

    # No negative samples
    y_true_no_neg = np.array([1] * 10)
    y_pred_no_neg = np.array([0] * 5 + [1] * 5)
    assert np.isnan(compute_fpr(y_true_no_neg, y_pred_no_neg))

    # No positive predictions
    y_true_some_pos = np.array([0] * 5 + [1] * 5)
    y_pred_no_pos_pred = np.array([0] * 10)
    assert np.isnan(compute_precision(y_true_some_pos, y_pred_no_pos_pred))
    assert np.isnan(compute_f1(y_true_some_pos, y_pred_no_pos_pred))

    print("Classification metrics verification passed.")
    return True


if __name__ == "__main__":
    verify_classification_metrics()
