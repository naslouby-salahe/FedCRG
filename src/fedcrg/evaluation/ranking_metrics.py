"""Ranking metrics independent of a threshold."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(average_precision_score(labels, scores))
