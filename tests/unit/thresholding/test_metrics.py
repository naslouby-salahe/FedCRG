from __future__ import annotations

import numpy as np

from fedcrg.thresholding.metrics import (
    attack_balanced_tpr,
    auprc,
    auroc,
    band_error,
    confusion_matrix,
    high_excess,
)
from fedcrg.types import OperatingBand

_BAND = OperatingBand(lower=0.005, upper=0.015)


def test_strict_threshold_operator_treats_equality_as_benign() -> None:
    cm = confusion_matrix(np.array([1.0, 1.1]), np.array([0, 1]), 1.0)
    assert cm.fp == 0
    assert cm.tp == 1
    assert cm.tn == 1
    assert cm.fn == 0


def test_operating_band_metrics() -> None:
    assert band_error(0.01, _BAND) == 0.0
    assert high_excess(0.02, _BAND) == 0.005000000000000001


def test_band_error_penalizes_below_band() -> None:
    assert band_error(0.0, _BAND) == 0.005


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
