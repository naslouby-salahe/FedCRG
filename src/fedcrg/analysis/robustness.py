"""Locked synthetic robustness generators from the pre-registration protocol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from fedcrg.core.enums import ExperimentAxisId
from fedcrg.core.types import OperatingBand
from fedcrg.protocol.readiness import ReadinessPlanBuilder


@dataclass(frozen=True, slots=True)
class RobustnessCell:
    axis: ExperimentAxisId
    value: float
    coverage: float
    repetitions: int


def _threshold(scores: np.ndarray, rank: int) -> float:
    return float(np.sort(scores, kind="stable")[rank - 1])


def temporal_dependence_stress(
    phi: float,
    sample_count: int,
    repetitions: int,
    band: OperatingBand,
    assurance: float,
    seed: int,
) -> RobustnessCell:
    if not -1.0 < phi < 1.0:
        raise ValueError("AR(1) phi must be strictly inside (-1,1)")
    plan = ReadinessPlanBuilder().build(sample_count, band, assurance)
    rng = np.random.Generator(np.random.PCG64(seed))
    inside = 0
    for _ in range(repetitions):
        innovations = rng.normal(size=sample_count)
        series = np.empty(sample_count, dtype=np.float64)
        series[0] = innovations[0]
        scale = np.sqrt(1.0 - phi**2)
        for index in range(1, sample_count):
            series[index] = phi * series[index - 1] + scale * innovations[index]
        fpr = 1.0 - norm.cdf(_threshold(series, plan.rank))
        inside += int(band.lower <= fpr <= band.upper)
    return RobustnessCell(
        ExperimentAxisId.PHI,
        phi,
        inside / repetitions,
        repetitions,
    )


def calibration_shift_stress(
    mean_shift: float,
    repetitions: int,
    band: OperatingBand,
    assurance: float,
    seed: int,
    sample_count: int = 2000,
) -> RobustnessCell:
    plan = ReadinessPlanBuilder().build(sample_count, band, assurance)
    rng = np.random.Generator(np.random.PCG64(seed))
    inside = 0
    for _ in range(repetitions):
        threshold = _threshold(rng.normal(size=sample_count), plan.rank)
        future_fpr = 1.0 - norm.cdf(threshold, loc=mean_shift, scale=1.0)
        inside += int(band.lower <= future_fpr <= band.upper)
    return RobustnessCell(
        ExperimentAxisId.MEAN_SHIFT,
        mean_shift,
        inside / repetitions,
        repetitions,
    )
