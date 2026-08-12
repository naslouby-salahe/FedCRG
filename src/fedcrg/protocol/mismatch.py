"""Independent evidence of material mismatch at the reference threshold."""

from __future__ import annotations

import math

import numpy as np
from scipy import special
from scipy.stats import binom

from fedcrg.core.enums import MismatchOutcome
from fedcrg.core.types import ConfidenceInterval, OperatingBand
from fedcrg.protocol.results import MismatchEvidence


def clopper_pearson_interval(x: int, n: int, confidence: float) -> ConfidenceInterval:
    if n <= 0 or not 0 <= x <= n:
        raise ValueError("Require n > 0 and 0 <= x <= n")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if x == 0 else float(special.betaincinv(x, n - x + 1, tail))
    upper = 1.0 if x == n else float(special.betaincinv(x + 1, n - x, 1.0 - tail))
    return ConfidenceInterval(lower=lower, upper=upper)


def minimum_mismatch_sample_count(lower_band: float, confidence: float) -> int:
    if lower_band <= 0.0:
        raise ValueError("A bidirectional mismatch check requires a positive lower band")
    tail = (1.0 - confidence) / 2.0
    estimate = max(1, int(math.floor(math.log(tail) / math.log(1.0 - lower_band))) + 1)
    while 1.0 - tail ** (1.0 / estimate) >= lower_band:
        estimate += 1
    while estimate > 1 and 1.0 - tail ** (1.0 / (estimate - 1)) < lower_band:
        estimate -= 1
    return estimate


class ReferenceMismatchEvaluator:
    def evaluate(
        self,
        scores: np.ndarray,
        reference_threshold: float,
        band: OperatingBand,
        confidence: float,
    ) -> MismatchEvidence:
        values = np.asarray(scores, dtype=np.float64)
        n = len(values)
        if n == 0:
            raise ValueError("Mismatch evidence requires at least one score")
        minimum = minimum_mismatch_sample_count(band.lower, confidence)
        if n < minimum:
            return MismatchEvidence(
                sample_count=n,
                exceedance_count=int(np.count_nonzero(values > reference_threshold)),
                estimated_fpr=float(np.mean(values > reference_threshold)),
                interval=None,
                outcome=MismatchOutcome.INSUFFICIENT_EVIDENCE,
                minimum_sample_count=minimum,
                p_low=None,
                p_high=None,
            )
        x = int(np.count_nonzero(values > reference_threshold))
        interval = clopper_pearson_interval(x, n, confidence)
        if interval.upper < band.lower:
            outcome = MismatchOutcome.LOW
        elif interval.lower > band.upper:
            outcome = MismatchOutcome.HIGH
        else:
            outcome = MismatchOutcome.NO_MATERIAL_DIFFERENCE
        return MismatchEvidence(
            sample_count=n,
            exceedance_count=x,
            estimated_fpr=x / n,
            interval=interval,
            outcome=outcome,
            minimum_sample_count=minimum,
            p_low=float(binom.cdf(x, n, band.lower)),
            p_high=float(binom.sf(x - 1, n, band.upper)),
        )
