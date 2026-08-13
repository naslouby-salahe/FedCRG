"""Strict-threshold classification metrics with explicit undefined-value semantics."""

from __future__ import annotations

from fedcrg.evaluation.confusion_matrix import ConfusionMatrix


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
