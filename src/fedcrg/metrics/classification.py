"""Strict-threshold classification metrics with explicit undefined-value semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    tp: int
    tn: int
    fp: int
    fn: int


def confusion_matrix(scores: np.ndarray, labels: np.ndarray, threshold: float) -> ConfusionMatrix:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.shape != targets.shape or values.ndim != 1:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")
    if not np.isfinite(values).all():
        raise ValueError("NONFINITE_SCORE")
    predictions = values > threshold
    positives = targets == 1
    negatives = targets == 0
    if np.count_nonzero(~(positives | negatives)):
        raise ValueError("Labels must be binary 0/1")
    return ConfusionMatrix(
        tp=int(np.count_nonzero(predictions & positives)),
        tn=int(np.count_nonzero(~predictions & negatives)),
        fp=int(np.count_nonzero(predictions & negatives)),
        fn=int(np.count_nonzero(~predictions & positives)),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def fpr(cm: ConfusionMatrix) -> float | None:
    return _ratio(cm.fp, cm.fp + cm.tn)


def tpr(cm: ConfusionMatrix) -> float | None:
    return _ratio(cm.tp, cm.tp + cm.fn)


def precision(cm: ConfusionMatrix) -> float | None:
    return _ratio(cm.tp, cm.tp + cm.fp)


def recall(cm: ConfusionMatrix) -> float | None:
    return tpr(cm)


def f1(cm: ConfusionMatrix) -> float | None:
    p = precision(cm)
    r = recall(cm)
    if p is None or r is None or p + r == 0.0:
        return None
    return 2.0 * p * r / (p + r)


def balanced_accuracy(cm: ConfusionMatrix) -> float | None:
    sensitivity = tpr(cm)
    specificity = _ratio(cm.tn, cm.tn + cm.fp)
    if sensitivity is None or specificity is None:
        return None
    return 0.5 * (sensitivity + specificity)
