import numpy as np

from fedcrg.evaluation.attack_balanced_metrics import attack_balanced_tpr
from fedcrg.evaluation.ranking_metrics import auprc, auroc


def test_ranking_metrics_are_perfect_for_separable_scores() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 1.0
    assert auprc(scores, labels) == 1.0


def test_attack_balanced_tpr_averages_groups() -> None:
    scores = np.array([0.1, 0.9, 0.6, 0.4, 0.8])
    labels = np.array([0, 1, 1, 1, 1])
    groups = np.array(["benign", "a", "a", "b", "b"])
    assert attack_balanced_tpr(scores, labels, groups, 0.5) == 0.75
