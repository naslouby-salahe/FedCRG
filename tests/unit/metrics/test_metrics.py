import numpy as np

from fedcrg.core.types import OperatingBand
from fedcrg.metrics.classification import confusion_matrix, fpr, tpr
from fedcrg.metrics.operating_band import band_error, high_excess


def test_strict_threshold_operator_treats_equality_as_benign() -> None:
    cm = confusion_matrix(np.array([1.0, 1.1]), np.array([0, 1]), 1.0)
    assert cm.fp == 0
    assert cm.tp == 1


def test_operating_band_metrics() -> None:
    band = OperatingBand(0.005, 0.015)
    assert band_error(0.01, band) == 0.0
    assert high_excess(0.02, band) == 0.005000000000000001
