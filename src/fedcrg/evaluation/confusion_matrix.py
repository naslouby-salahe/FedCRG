"""Strict-threshold confusion-matrix computation."""

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
