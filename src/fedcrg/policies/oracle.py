"""Final-test oracle used only as an unattainable diagnostic ceiling."""

from __future__ import annotations

import numpy as np

from fedcrg.domain.values import OperatingBand
from fedcrg.metrics.classification import confusion_matrix, tpr
from fedcrg.metrics.operating_band import band_error
from fedcrg.policies.base import FinalTestEvidence


def oracle_choice(
    client: FinalTestEvidence,
    candidates: tuple[float, float, float],
    band: OperatingBand,
) -> float:
    """Choose minimum band error, then higher TPR, then locked candidate order."""

    ranked: list[tuple[float, float, int, float]] = []
    benign_labels = np.zeros(len(client.benign_test_scores), dtype=np.int64)
    attack_labels = np.ones(len(client.attack_test_scores), dtype=np.int64)
    for order, threshold in enumerate(candidates):
        benign_cm = confusion_matrix(client.benign_test_scores, benign_labels, threshold)
        attack_cm = confusion_matrix(client.attack_test_scores, attack_labels, threshold)
        client_fpr = benign_cm.fp / (benign_cm.fp + benign_cm.tn)
        client_tpr = tpr(attack_cm)
        tpr_rank = -1.0 if client_tpr is None else client_tpr
        ranked.append((band_error(client_fpr, band), -tpr_rank, order, threshold))
    return min(ranked)[3]
