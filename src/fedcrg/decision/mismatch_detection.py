"""Exact benign reference-mismatch evidence and fleet-level sensitivities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import special
from scipy.stats import binom

from fedcrg.domain.enums import MismatchOutcome
from fedcrg.domain.identifiers import ClientId
from fedcrg.domain.values import ConfidenceInterval, OperatingBand
from fedcrg.decision.results import MismatchEvidence


def clopper_pearson_interval(x: int, n: int, confidence: float) -> ConfidenceInterval:
    if n <= 0 or not 0 <= x <= n:
        raise ValueError("Require n > 0 and 0 <= x <= n")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if x == 0 else float(special.betaincinv(x, n - x + 1, tail))
    upper = 1.0 if x == n else float(special.betaincinv(x + 1, n - x, 1.0 - tail))
    return ConfidenceInterval(lower=lower, upper=upper)


def minimum_bidirectional_sample_count(lower_band: float, confidence: float) -> int | None:
    if not 0.0 <= lower_band < 1.0:
        raise ValueError("lower_band must be in [0, 1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if lower_band == 0.0:
        return None
    tail = (1.0 - confidence) / 2.0
    estimate = max(
        1,
        int(math.floor(math.log(tail) / math.log(1.0 - lower_band))) + 1,
    )
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
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("Mismatch evidence requires a non-empty one-dimensional array")
        if not np.isfinite(values).all() or not math.isfinite(reference_threshold):
            raise ValueError("Mismatch scores and threshold must be finite")

        n = len(values)
        x = int(np.count_nonzero(values > reference_threshold))
        interval = clopper_pearson_interval(x, n, confidence)
        minimum = minimum_bidirectional_sample_count(band.lower, confidence)
        high_side_only = minimum is None

        if minimum is not None and n < minimum:
            outcome = MismatchOutcome.INSUFFICIENT_EVIDENCE
        elif not high_side_only and interval.upper < band.lower:
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
            p_low=(None if high_side_only else float(binom.cdf(x, n, band.lower))),
            p_high=float(binom.sf(x - 1, n, band.upper)),
            high_side_only=high_side_only,
        )


@dataclass(frozen=True, slots=True)
class FleetMismatchDecision:
    client_id: ClientId
    outcome: MismatchOutcome
    low_p_value: float | None
    high_p_value: float


def bonferroni_fleet_sensitivity(
    counts: dict[ClientId, tuple[int, int]],
    band: OperatingBand,
    *,
    familywise_alpha: float,
) -> tuple[FleetMismatchDecision, ...]:
    if not counts:
        return ()
    confidence = 1.0 - familywise_alpha / len(counts)
    decisions: list[FleetMismatchDecision] = []
    for client_id in sorted(counts):
        x, n = counts[client_id]
        interval = clopper_pearson_interval(x, n, confidence)
        outcome = MismatchOutcome.NO_MATERIAL_DIFFERENCE
        if band.lower > 0.0 and interval.upper < band.lower:
            outcome = MismatchOutcome.LOW
        elif interval.lower > band.upper:
            outcome = MismatchOutcome.HIGH
        decisions.append(
            FleetMismatchDecision(
                client_id=client_id,
                outcome=outcome,
                low_p_value=(None if band.lower == 0.0 else float(binom.cdf(x, n, band.lower))),
                high_p_value=float(binom.sf(x - 1, n, band.upper)),
            )
        )
    return tuple(decisions)


def _holm_rejected(
    hypotheses: list[tuple[float, tuple[ClientId, MismatchOutcome]]],
    alpha: float,
) -> set[tuple[ClientId, MismatchOutcome]]:
    rejected: set[tuple[ClientId, MismatchOutcome]] = set()
    ordered = sorted(hypotheses, key=lambda item: (item[0], item[1][0], item[1][1].value))
    total = len(ordered)
    for index, (p_value, key) in enumerate(ordered):
        threshold = alpha / (total - index)
        if p_value <= threshold:
            rejected.add(key)
        else:
            break
    return rejected


def holm_directional_fleet_sensitivity(
    counts: dict[ClientId, tuple[int, int]],
    band: OperatingBand,
    *,
    familywise_alpha: float,
) -> tuple[FleetMismatchDecision, ...]:
    hypotheses: list[tuple[float, tuple[ClientId, MismatchOutcome]]] = []
    diagnostics: dict[ClientId, tuple[float | None, float]] = {}
    for client_id in sorted(counts):
        x, n = counts[client_id]
        low = None if band.lower == 0.0 else float(binom.cdf(x, n, band.lower))
        high = float(binom.sf(x - 1, n, band.upper))
        diagnostics[client_id] = (low, high)
        if low is not None:
            hypotheses.append((low, (client_id, MismatchOutcome.LOW)))
        hypotheses.append((high, (client_id, MismatchOutcome.HIGH)))

    rejected = _holm_rejected(hypotheses, familywise_alpha)
    decisions: list[FleetMismatchDecision] = []
    for client_id in sorted(counts):
        low_rejected = (client_id, MismatchOutcome.LOW) in rejected
        high_rejected = (client_id, MismatchOutcome.HIGH) in rejected
        if low_rejected and high_rejected:
            raise RuntimeError(f"DIRECTION_CONTRADICTION: both directions rejected for {client_id}")
        outcome = (
            MismatchOutcome.LOW
            if low_rejected
            else MismatchOutcome.HIGH
            if high_rejected
            else MismatchOutcome.NO_MATERIAL_DIFFERENCE
        )
        low, high = diagnostics[client_id]
        decisions.append(FleetMismatchDecision(client_id, outcome, low, high))
    return tuple(decisions)
