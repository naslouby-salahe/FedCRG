"""Final-test oracle used only as an unattainable diagnostic ceiling."""

from __future__ import annotations

import numpy as np

from fedcrg.core.types import OperatingBand
from fedcrg.metrics.classification import confusion_matrix, tpr
from fedcrg.metrics.operating_band import band_error
from fedcrg.policies.base import FinalTestEvidence


def oracle_choice(
    client: FinalTestEvidence,
    candidates: tuple[float, float, float],
    band: OperatingBand,
) -> float:
    ranked: list[tuple[float, float, int, float]] = []
    benign_labels = np.zeros(len(client.benign_test_scores), dtype=np.int64)
    attack_labels = np.ones(len(client.attack_test_scores), dtype=np.int64)
    for order, threshold in enumerate(candidates):
        benign_cm = confusion_matrix(client.benign_test_scores, benign_labels, threshold)
        attack_cm = confusion_matrix(client.attack_test_scores, attack_labels, threshold)
        client_fpr = benign_cm.fp / (benign_cm.fp + benign_cm.tn)
        client_tpr = tpr(attack_cm)
        ranked.append((band_error(client_fpr, band), -(client_tpr or -1.0), order, threshold))
    return min(ranked)[3]
