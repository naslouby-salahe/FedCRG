from __future__ import annotations

import numpy as np

from fedcrg.thresholding.metrics import band_error, confusion_matrix, high_excess
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
