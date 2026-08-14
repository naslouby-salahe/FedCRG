"""Finite-sample operating-point governance: the federation reference
threshold, calibration readiness with the realized local threshold, reference
mismatch evidence, and the five-state deployment decision.

Observed score values never participate in rank optimization. The readiness
plan is keyed only by the pre-data statistical contract ``(n, a, b,
assurance)``; the realized calibration scores enter only when selecting the
r*-th order statistic after the rank has been fixed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from scipy import special
from scipy.stats import binom

from fedcrg.types import (
    Alpha,
    Identifier,
    JsonValue,
    PlanKey,
    Assurance,
    BinomialCounts,
    CalibrationReadinessState,
    ClientId,
    ConfidenceInterval,
    ConfidenceLevel,
    CoverageProbability,
    DecisionReason,
    DecisionState,
    ExceedanceCount,
    Fpr,
    Fraction,
    MismatchOutcome,
    OperatingBand,
    PValue,
    PositiveCount,
    Probability,
    SampleCount,
    Spacing,
    Threshold,
    ThresholdSource,
    TIGHT_TOLERANCE,
)

Frozen = ConfigDict(frozen=True)

class ReferenceThreshold(BaseModel):
    """Federation-wide reference operating point from equal-count pooled R."""

    model_config = Frozen

    value: Threshold
    rank: PositiveCount
    sample_count: PositiveCount
    client_count: PositiveCount
    samples_per_client: PositiveCount

class ReadinessPlan(BaseModel):
    """Pre-data order-statistic plan for one ``(n, a, b, assurance)`` contract."""

    model_config = Frozen

    sample_count: SampleCount
    rank: PositiveCount
    coverage_probability: CoverageProbability
    state: CalibrationReadinessState
    band: OperatingBand
    assurance: Assurance

class ContinuityDiagnostics(BaseModel):
    """Score-shape diagnostics at the selected local order statistic."""

    model_config = Frozen

    unique_score_fraction: Fraction
    duplicate_count: ExceedanceCount
    selected_threshold_multiplicity: ExceedanceCount
    minimum_positive_spacing: Spacing | None = None

class CalibrationReadiness(BaseModel):
    """Readiness outcome for one client: the realized local threshold and ties."""

    model_config = Frozen

    plan: ReadinessPlan
    threshold: Threshold | None = None
    diagnostics: ContinuityDiagnostics

    @property
    def tie_count(self) -> ExceedanceCount:
        return self.diagnostics.selected_threshold_multiplicity

class MismatchEvidence(BaseModel):
    """Exact-binomial evidence that the reference threshold is materially wrong."""

    model_config = Frozen

    sample_count: SampleCount
    exceedance_count: ExceedanceCount
    estimated_fpr: Fpr
    interval: ConfidenceInterval
    outcome: MismatchOutcome
    minimum_sample_count: SampleCount | None = None
    p_low: PValue | None = None
    p_high: PValue
    high_side_only: bool = False

class ThresholdDecision(BaseModel):
    """Final deployment decision for one client."""

    model_config = Frozen

    state: DecisionState
    threshold: Threshold
    source: ThresholdSource
    reason: DecisionReason
    tie_count: ExceedanceCount

class ClientEvaluationResult(BaseModel):
    """Complete per-client governance evidence bundle."""

    model_config = Frozen

    client_id: ClientId
    reference: ReferenceThreshold
    readiness: CalibrationReadiness
    mismatch: MismatchEvidence
    decision: ThresholdDecision

def reference_rank(sample_count: SampleCount, alpha: Alpha) -> PositiveCount:
    """q_ref = min(N_R, ceil((N_R + 1)(1 - alpha)))."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return min(sample_count, math.ceil((sample_count + 1) * (1.0 - alpha)))

def build_reference_threshold(
    scores_by_client: Mapping[ClientId, np.ndarray],
    alpha: Alpha,
) -> ReferenceThreshold:
    """Pool equal-count per-client reference scores and take the locked rank."""
    if not scores_by_client:
        raise ValueError("At least one client must contribute reference scores")
    lengths = {len(np.asarray(scores)) for scores in scores_by_client.values()}
    if 0 in lengths:
        raise ValueError("Reference score arrays must be non-empty")
    if len(lengths) != 1:
        raise ValueError("Each client must contribute the same number of reference scores")
    samples_per_client = next(iter(lengths))
    pooled = np.concatenate(
        [np.asarray(scores, dtype=np.float64) for scores in scores_by_client.values()]
    )
    rank = reference_rank(len(pooled), alpha)
    threshold = float(np.sort(pooled, kind="stable")[rank - 1])
    return ReferenceThreshold(
        value=threshold,
        rank=rank,
        sample_count=len(pooled),
        client_count=len(scores_by_client),
        samples_per_client=samples_per_client,
    )

def coverage_probability(rank: PositiveCount, sample_count: SampleCount, band: OperatingBand) -> CoverageProbability:
    """P_r = I_b(n+1-r, r) - I_a(n+1-r, r) in float64."""
    if not 1 <= rank <= sample_count:
        raise ValueError("rank must be inside [1, sample_count]")
    upper_shape = sample_count + 1 - rank
    lower_shape = rank
    probability = special.betainc(upper_shape, lower_shape, band.upper)
    probability -= special.betainc(upper_shape, lower_shape, band.lower)
    return float(probability)

class ReadinessPlanBuilder:
    """Choose the order-statistic rank using protocol constants only."""

    def build(self, sample_count: SampleCount, band: OperatingBand, assurance: Assurance) -> ReadinessPlan:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if not 0.0 < assurance < 1.0:
            raise ValueError("assurance must be in (0,1)")
        best_rank = 1
        best_probability = -1.0
        for rank in range(1, sample_count + 1):
            probability = coverage_probability(rank, sample_count, band)
            if probability > best_probability or (
                math.isclose(probability, best_probability, rel_tol=0.0, abs_tol=1e-15)
                and rank > best_rank
            ):
                best_rank = rank
                best_probability = probability
        state = (
            CalibrationReadinessState.READY
            if best_probability >= assurance
            else CalibrationReadinessState.NOT_READY
        )
        return ReadinessPlan(
            sample_count=sample_count,
            rank=best_rank,
            coverage_probability=best_probability,
            state=state,
            band=band,
            assurance=assurance,
        )

class ReadinessPlanCache:
    """Persistent pre-data plan table consumed by real-data policy evaluation."""

    def __init__(
        self,
        path: Path | None = None,
        builder: ReadinessPlanBuilder | None = None,
    ) -> None:
        self.path = path
        self.builder = builder or ReadinessPlanBuilder()
        self._plans: dict[PlanKey, ReadinessPlan] = {}
        if path is not None and path.is_file():
            self.load(path)

    def __len__(self) -> int:
        return len(self._plans)

    @staticmethod
    def key(sample_count: SampleCount, band: OperatingBand, assurance: Assurance) -> PlanKey:
        return (
            f"n={sample_count}|a={band.lower:.17g}|b={band.upper:.17g}|assurance={assurance:.17g}"
        )

    def precompute(self, sample_count: SampleCount, band: OperatingBand, assurance: Assurance) -> ReadinessPlan:
        key = self.key(sample_count, band, assurance)
        candidate = self.builder.build(sample_count, band, assurance)
        existing = self._plans.get(key)
        if existing is not None and existing != candidate:
            raise RuntimeError("Frozen readiness-plan table is internally inconsistent")
        self._plans[key] = candidate
        return candidate

    def require(self, sample_count: SampleCount, band: OperatingBand, assurance: Assurance) -> ReadinessPlan:
        key = self.key(sample_count, band, assurance)
        try:
            return self._plans[key]
        except KeyError as exc:
            raise FileNotFoundError(
                "Required pre-data readiness plan is absent. Run the protocol "
                "precomputation command before evaluating real scores: " + key
            ) from exc

    def plans(self) -> tuple[ReadinessPlan, ...]:
        return tuple(plan for _, plan in sorted(self._plans.items()))

    def load_plans(self, plans: tuple[ReadinessPlan, ...]) -> None:
        for plan in plans:
            expected_key = self.key(plan.sample_count, plan.band, plan.assurance)
            regenerated = self.builder.build(plan.sample_count, plan.band, plan.assurance)
            if (
                regenerated.rank != plan.rank
                or regenerated.state is not plan.state
                or abs(regenerated.coverage_probability - plan.coverage_probability)
                > TIGHT_TOLERANCE
            ):
                raise ValueError("Readiness-plan table failed formula regeneration")
            self._plans[expected_key] = plan

    def save(self, path: Path | None = None) -> None:
        """Persist the frozen plan table atomically."""
        from fedcrg.evidence.store import atomic_write_json

        target = path or self.path
        if target is None:
            raise ValueError("Readiness-plan cache has no persistence path")
        atomic_write_json(
            target,
            [plan.model_dump(mode="json") for plan in self.plans()],
        )

    def load(self, path: Path) -> None:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Readiness-plan table must be a JSON array")
        self.load_plans(tuple(ReadinessPlan.model_validate(entry) for entry in raw))

class CalibrationReadinessEvaluator:
    """Select the precomputed order statistic and audit continuity at that point."""

    def evaluate(self, scores: np.ndarray, plan: ReadinessPlan) -> CalibrationReadiness:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or len(values) != plan.sample_count:
            raise ValueError("Observed calibration size does not match the frozen readiness plan")
        if not np.isfinite(values).all():
            raise ValueError("Calibration scores must be finite")
        diagnostics = continuity_diagnostics(values, plan.rank)
        if plan.state is CalibrationReadinessState.NOT_READY:
            return CalibrationReadiness(plan=plan, threshold=None, diagnostics=diagnostics)
        ordered = np.sort(values, kind="stable")
        threshold = float(ordered[plan.rank - 1])
        return CalibrationReadiness(plan=plan, threshold=threshold, diagnostics=diagnostics)

def continuity_diagnostics(scores: np.ndarray, selected_rank: PositiveCount) -> ContinuityDiagnostics:
    """Tie and continuity diagnostics at one rank."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Continuity diagnostics require a non-empty score vector")
    ordered = np.sort(values, kind="stable")
    if not 1 <= selected_rank <= len(ordered):
        raise ValueError("selected_rank must lie inside the score vector")
    selected = float(ordered[selected_rank - 1])
    unique = np.unique(ordered)
    duplicate_count = int(len(ordered) - len(unique))
    positive_spacing = np.diff(unique)
    minimum_spacing = None if len(positive_spacing) == 0 else float(np.min(positive_spacing))
    return ContinuityDiagnostics(
        unique_score_fraction=float(len(unique) / len(ordered)),
        duplicate_count=duplicate_count,
        selected_threshold_multiplicity=int(np.count_nonzero(ordered == selected)),
        minimum_positive_spacing=minimum_spacing,
    )

def familywise_readiness_assurance(client_count: PositiveCount, familywise_alpha: Probability) -> Assurance:
    """Bonferroni per-client assurance 1 - alpha/K for a familywise target."""
    if client_count <= 0:
        raise ValueError("client_count must be positive")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must be in (0,1)")
    return 1.0 - familywise_alpha / client_count

def clopper_pearson_interval(counts: BinomialCounts, confidence: ConfidenceLevel) -> ConfidenceInterval:
    """Exact two-sided Clopper-Pearson interval for the benign exceedance rate."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    tail = (1.0 - confidence) / 2.0
    lower = (
        0.0
        if counts.x == 0
        else float(special.betaincinv(counts.x, counts.n - counts.x + 1, tail))
    )
    upper = (
        1.0
        if counts.x == counts.n
        else float(special.betaincinv(counts.x + 1, counts.n - counts.x, 1.0 - tail))
    )
    return ConfidenceInterval(lower=lower, upper=upper)

def minimum_bidirectional_sample_count(lower_band: Fpr, confidence: ConfidenceLevel) -> SampleCount | None:
    """Smallest n with 1 - ((1-confidence)/2)^(1/n) < a; None when a == 0."""
    if not 0.0 <= lower_band < 1.0:
        raise ValueError("lower_band must be in [0, 1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if lower_band == 0.0:
        return None
    tail = (1.0 - confidence) / 2.0
    estimate = max(1, int(math.floor(math.log(tail) / math.log(1.0 - lower_band))) + 1)
    while 1.0 - tail ** (1.0 / estimate) >= lower_band:
        estimate += 1
    while estimate > 1 and 1.0 - tail ** (1.0 / (estimate - 1)) < lower_band:
        estimate -= 1
    return estimate

class ReferenceMismatchEvaluator:
    """Exact reference-mismatch evidence evaluator."""
    def evaluate(
        self,
        scores: np.ndarray,
        reference_threshold: Threshold,
        band: OperatingBand,
        confidence: ConfidenceLevel,
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

class FleetMismatchDecision(BaseModel):
    """Fleet-level mismatch decision for one client."""
    model_config = Frozen

    client_id: ClientId
    outcome: MismatchOutcome
    low_p_value: PValue | None = None
    high_p_value: PValue

class DirectionalHypothesis(BaseModel):
    """One directional mismatch hypothesis: client + claimed direction."""

    model_config = Frozen

    client_id: ClientId
    outcome: MismatchOutcome
    p_value: PValue

def _directional_p_values(
    counts: BinomialCounts, band: OperatingBand
) -> tuple[PValue | None, PValue]:
    low = None if band.lower == 0.0 else float(binom.cdf(counts.x, counts.n, band.lower))
    high = float(binom.sf(counts.x - 1, counts.n, band.upper))
    return low, high

def bonferroni_fleet_sensitivity(
    counts_by_client: Mapping[ClientId, BinomialCounts],
    band: OperatingBand,
    *,
    familywise_alpha: Probability,
) -> tuple[FleetMismatchDecision, ...]:
    """Bonferroni fleet-level sensitivity analysis."""
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
    alpha: Probability,
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
    counts_by_client: Mapping[ClientId, BinomialCounts],
    band: OperatingBand,
    *,
    familywise_alpha: Probability,
) -> tuple[FleetMismatchDecision, ...]:
    """Holm-Bonferroni directional fleet sensitivity."""
    hypotheses: list[DirectionalHypothesis] = []
    diagnostics: dict[ClientId, tuple[float | None, float]] = {}
    for client_id in sorted(counts_by_client):
        low, high = _directional_p_values(counts_by_client[client_id], band)
        diagnostics[client_id] = (low, high)
        if low is not None:
            hypotheses.append(
                DirectionalHypothesis(
                    client_id=client_id, outcome=MismatchOutcome.LOW, p_value=low
                )
            )
        hypotheses.append(
            DirectionalHypothesis(
                client_id=client_id, outcome=MismatchOutcome.HIGH, p_value=high
            )
        )

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
        decisions.append(
            FleetMismatchDecision(
                client_id=client_id,
                outcome=outcome,
                low_p_value=low,
                high_p_value=high,
            )
        )
    return tuple(decisions)

class DeploymentDecision:
    """Combine independent evidence without reinterpreting inconclusive states."""

    def decide(
        self,
        reference: ReferenceThreshold,
        readiness: CalibrationReadiness,
        mismatch: MismatchEvidence,
        reject_calibration_ties: bool,
    ) -> ThresholdDecision:
        tie_count = readiness.tie_count
        if mismatch.outcome is MismatchOutcome.INSUFFICIENT_EVIDENCE:
            return ThresholdDecision(
                state=DecisionState.MISMATCH_EVIDENCE_INSUFFICIENT,
                threshold=reference.value,
                source=ThresholdSource.REFERENCE,
                reason=DecisionReason.INSUFFICIENT_MISMATCH_EVIDENCE,
                tie_count=tie_count,
            )
        if mismatch.outcome is MismatchOutcome.NO_MATERIAL_DIFFERENCE:
            return ThresholdDecision(
                state=DecisionState.REFERENCE_RETAINED,
                threshold=reference.value,
                source=ThresholdSource.REFERENCE,
                reason=DecisionReason.NO_MATERIAL_DIFFERENCE,
                tie_count=tie_count,
            )
        if readiness.plan.state is CalibrationReadinessState.NOT_READY or readiness.threshold is None:
            return ThresholdDecision(
                state=DecisionState.CALIBRATION_DEFICIT,
                threshold=reference.value,
                source=ThresholdSource.REFERENCE,
                reason=DecisionReason.CALIBRATION_NOT_READY,
                tie_count=tie_count,
            )
        if reject_calibration_ties and tie_count > 1:
            return ThresholdDecision(
                state=DecisionState.ASSUMPTION_VIOLATION,
                threshold=reference.value,
                source=ThresholdSource.REFERENCE,
                reason=DecisionReason.CALIBRATION_TIE,
                tie_count=tie_count,
            )
        return ThresholdDecision(
            state=DecisionState.PERSONALIZED,
            threshold=readiness.threshold,
            source=ThresholdSource.LOCAL_CALIBRATION,
            reason=DecisionReason.LOCAL_PERSONALIZATION_ADMITTED,
            tie_count=tie_count,
        )
