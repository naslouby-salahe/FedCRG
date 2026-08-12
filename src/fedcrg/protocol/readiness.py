"""Calibration-readiness planning and evaluation."""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from fedcrg.core.enums import CalibrationReadinessState
from fedcrg.core.types import OperatingBand
from fedcrg.protocol.results import CalibrationReadiness, ReadinessPlan


def coverage_probability(rank: int, sample_count: int, band: OperatingBand) -> float:
    if not 1 <= rank <= sample_count:
        raise ValueError("rank must be inside [1, sample_count]")
    shape_a = sample_count + 1 - rank
    shape_b = rank
    return float(special.betainc(shape_a, shape_b, band.upper) - special.betainc(shape_a, shape_b, band.lower))


class CalibrationReadinessPlanner:
    """Plans the order statistic before observed score values are inspected."""

    def plan(self, sample_count: int, band: OperatingBand, assurance: float) -> ReadinessPlan:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if not 0.0 < assurance < 1.0:
            raise ValueError("assurance must be in (0, 1)")
        best_rank = 1
        best_probability = -1.0
        for rank in range(1, sample_count + 1):
            probability = coverage_probability(rank, sample_count, band)
            if probability > best_probability or (math.isclose(probability, best_probability, rel_tol=0.0, abs_tol=1e-15) and rank > best_rank):
                best_rank = rank
                best_probability = probability
        state = CalibrationReadinessState.READY if best_probability >= assurance else CalibrationReadinessState.NOT_READY
        return ReadinessPlan(sample_count=sample_count, rank=best_rank, coverage_probability=best_probability, state=state, band=band, assurance=assurance)


class CalibrationReadinessEvaluator:
    def evaluate(self, scores: np.ndarray, plan: ReadinessPlan) -> CalibrationReadiness:
        values = np.asarray(scores, dtype=np.float64)
        if len(values) != plan.sample_count:
            raise ValueError("Observed calibration size does not match the precomputed plan")
        if plan.state is CalibrationReadinessState.NOT_READY:
            return CalibrationReadiness(plan=plan, threshold=None, tie_count=0)
        ordered = np.sort(values, kind="stable")
        threshold = float(ordered[plan.rank - 1])
        tie_count = int(np.count_nonzero(ordered == threshold))
        return CalibrationReadiness(plan=plan, threshold=threshold, tie_count=tie_count)
