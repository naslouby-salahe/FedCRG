"""Exact benign reference-mismatch evidence and fleet-level sensitivities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import special
from scipy.stats import binom

from fedcrg.domain.enums import MismatchOutcome
from fedcrg.domain.identifiers import ClientId
from fedcrg.domain.values import BinomialCounts, ConfidenceInterval, OperatingBand
from fedcrg.decision.results import MismatchEvidence


def clopper_pearson_interval(counts: BinomialCounts, confidence: float) -> ConfidenceInterval:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    tail = (1.0 - confidence) / 2.0
    lower = (
        0.0 if counts.x == 0 else float(special.betaincinv(counts.x, counts.n - counts.x + 1, tail))
    )
    upper = (
        1.0
        if counts.x == counts.n
        else float(special.betaincinv(counts.x + 1, counts.n - counts.x, 1.0 - tail))
    )
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

        counts = BinomialCounts(
            x=int(np.count_nonzero(values > reference_threshold)), n=len(values)
        )
        interval = clopper_pearson_interval(counts, confidence)
        minimum = minimum_bidirectional_sample_count(band.lower, confidence)
        high_side_only = minimum is None

        if minimum is not None and counts.n < minimum:
            outcome = MismatchOutcome.INSUFFICIENT_EVIDENCE
        elif not high_side_only and interval.upper < band.lower:
            outcome = MismatchOutcome.LOW
        elif interval.lower > band.upper:
            outcome = MismatchOutcome.HIGH
        else:
            outcome = MismatchOutcome.NO_MATERIAL_DIFFERENCE

        return MismatchEvidence(
            sample_count=counts.n,
            exceedance_count=counts.x,
            estimated_fpr=counts.exceedance_rate,
            interval=interval,
            outcome=outcome,
            minimum_sample_count=minimum,
            p_low=(None if high_side_only else float(binom.cdf(counts.x, counts.n, band.lower))),
            p_high=float(binom.sf(counts.x - 1, counts.n, band.upper)),
            high_side_only=high_side_only,
        )


@dataclass(frozen=True, slots=True)
class FleetMismatchDecision:
    client_id: ClientId
    outcome: MismatchOutcome
    low_p_value: float | None
    high_p_value: float


@dataclass(frozen=True, slots=True)
class DirectionalHypothesis:
    """One directional mismatch hypothesis: client + claimed direction."""

    client_id: ClientId
    outcome: MismatchOutcome
    p_value: float


def _directional_p_values(
    counts: BinomialCounts, band: OperatingBand
) -> tuple[float | None, float]:
    low = None if band.lower == 0.0 else float(binom.cdf(counts.x, counts.n, band.lower))
    high = float(binom.sf(counts.x - 1, counts.n, band.upper))
    return low, high


def bonferroni_fleet_sensitivity(
    counts_by_client: dict[ClientId, BinomialCounts],
    band: OperatingBand,
    *,
    familywise_alpha: float,
) -> tuple[FleetMismatchDecision, ...]:
    if not counts_by_client:
        return ()
    confidence = 1.0 - familywise_alpha / len(counts_by_client)
    decisions: list[FleetMismatchDecision] = []
    for client_id in sorted(counts_by_client):
        counts = counts_by_client[client_id]
        interval = clopper_pearson_interval(counts, confidence)
        outcome = MismatchOutcome.NO_MATERIAL_DIFFERENCE
        if band.lower > 0.0 and interval.upper < band.lower:
            outcome = MismatchOutcome.LOW
        elif interval.lower > band.upper:
            outcome = MismatchOutcome.HIGH
        low, high = _directional_p_values(counts, band)
        decisions.append(
            FleetMismatchDecision(
                client_id=client_id,
                outcome=outcome,
                low_p_value=low,
                high_p_value=high,
            )
        )
    return tuple(decisions)


def _holm_rejected(
    hypotheses: list[DirectionalHypothesis],
    alpha: float,
) -> set[tuple[ClientId, MismatchOutcome]]:
    rejected: set[tuple[ClientId, MismatchOutcome]] = set()
    ordered = sorted(
        hypotheses,
        key=lambda item: (item.p_value, item.client_id, item.outcome.value),
    )
    total = len(ordered)
    for index, hypothesis in enumerate(ordered):
        threshold = alpha / (total - index)
        if hypothesis.p_value <= threshold:
            rejected.add((hypothesis.client_id, hypothesis.outcome))
        else:
            break
    return rejected


def holm_directional_fleet_sensitivity(
    counts_by_client: dict[ClientId, BinomialCounts],
    band: OperatingBand,
    *,
    familywise_alpha: float,
) -> tuple[FleetMismatchDecision, ...]:
    hypotheses: list[DirectionalHypothesis] = []
    diagnostics: dict[ClientId, tuple[float | None, float]] = {}
    for client_id in sorted(counts_by_client):
        low, high = _directional_p_values(counts_by_client[client_id], band)
        diagnostics[client_id] = (low, high)
        if low is not None:
            hypotheses.append(DirectionalHypothesis(client_id, MismatchOutcome.LOW, low))
        hypotheses.append(DirectionalHypothesis(client_id, MismatchOutcome.HIGH, high))

    rejected = _holm_rejected(hypotheses, familywise_alpha)
    decisions: list[FleetMismatchDecision] = []
    for client_id in sorted(counts_by_client):
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
