from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy import special
from scipy.stats import binom

from fedcrg.types import (
    Alpha,
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
    model_config = Frozen

    value: Threshold
    rank: PositiveCount
    sample_count: PositiveCount
    client_count: PositiveCount
    samples_per_client: PositiveCount


class ReadinessPlan(BaseModel):
    model_config = Frozen

    sample_count: SampleCount
    rank: PositiveCount
    coverage_probability: CoverageProbability
    state: CalibrationReadinessState
    band: OperatingBand
    assurance: Assurance


class ContinuityDiagnostics(BaseModel):
    model_config = Frozen

    unique_score_fraction: Fraction
    duplicate_count: ExceedanceCount
    selected_threshold_multiplicity: ExceedanceCount
    minimum_positive_spacing: Spacing | None = None


class CalibrationReadiness(BaseModel):
    model_config = Frozen

    plan: ReadinessPlan
    threshold: Threshold | None = None
    diagnostics: ContinuityDiagnostics

    @property
    def tie_count(self) -> ExceedanceCount:
        return self.diagnostics.selected_threshold_multiplicity


class MismatchEvidence(BaseModel):
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
    model_config = Frozen

    state: DecisionState
    threshold: Threshold
    source: ThresholdSource
    reason: DecisionReason
    tie_count: ExceedanceCount


class ClientEvaluationResult(BaseModel):
    model_config = Frozen

    client_id: ClientId
    reference: ReferenceThreshold
    readiness: CalibrationReadiness
    mismatch: MismatchEvidence
    decision: ThresholdDecision


def reference_rank(sample_count: SampleCount, alpha: Alpha) -> PositiveCount:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return min(sample_count, math.ceil((sample_count + 1) * (1.0 - alpha)))


def build_reference_threshold(
    scores_by_client: Mapping[ClientId, np.ndarray],
    alpha: Alpha,
) -> ReferenceThreshold:
    if not scores_by_client:
        raise ValueError("At least one client must contribute reference scores")

    arrays = [np.asarray(scores, dtype=np.float64) for scores in scores_by_client.values()]
    samples_per_client = arrays[0].size

    if samples_per_client == 0:
        raise ValueError("Reference score arrays must be non-empty")
    if any(arr.size != samples_per_client for arr in arrays):
        raise ValueError("Each client must contribute the same number of reference scores")

    pooled = np.concatenate(arrays)
    sample_count = pooled.size
    rank = reference_rank(sample_count, alpha)
    threshold = float(np.sort(pooled, kind="stable")[rank - 1])

    return ReferenceThreshold(
        value=threshold,
        rank=rank,
        sample_count=sample_count,
        client_count=len(arrays),
        samples_per_client=samples_per_client,
    )


def coverage_probability(
    rank: PositiveCount, sample_count: SampleCount, band: OperatingBand
) -> CoverageProbability:
    if not 1 <= rank <= sample_count:
        raise ValueError("rank must be inside [1, sample_count]")
    upper_shape = sample_count + 1 - rank
    lower_shape = rank
    probability = special.betainc(upper_shape, lower_shape, band.upper)
    probability -= special.betainc(upper_shape, lower_shape, band.lower)
    return float(probability)


class ReadinessPlanBuilder:
    def build(
        self, sample_count: SampleCount, band: OperatingBand, assurance: Assurance
    ) -> ReadinessPlan:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if not 0.0 < assurance < 1.0:
            raise ValueError("assurance must be in (0,1)")

        ranks = np.arange(1, sample_count + 1)
        upper_shapes = sample_count + 1 - ranks
        
        probs = (
            special.betainc(upper_shapes, ranks, band.upper) - 
            special.betainc(upper_shapes, ranks, band.lower)
        )

        max_prob = np.max(probs)
        close_mask = np.isclose(probs, max_prob, rtol=0.0, atol=1e-15)
        best_rank = int(np.max(ranks[close_mask]))
        best_probability = float(probs[best_rank - 1])

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

    def precompute(
        self, sample_count: SampleCount, band: OperatingBand, assurance: Assurance
    ) -> ReadinessPlan:
        key = self.key(sample_count, band, assurance)
        candidate = self.builder.build(sample_count, band, assurance)
        existing = self._plans.get(key)
        if existing is not None and existing != candidate:
            raise RuntimeError("Frozen readiness-plan table is internally inconsistent")
        self._plans[key] = candidate
        return candidate

    def require(
        self, sample_count: SampleCount, band: OperatingBand, assurance: Assurance
    ) -> ReadinessPlan:
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
    def evaluate(self, scores: np.ndarray, plan: ReadinessPlan) -> CalibrationReadiness:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or values.size != plan.sample_count:
            raise ValueError("Observed calibration size does not match the frozen readiness plan")
        if not np.isfinite(values).all():
            raise ValueError("Calibration scores must be finite")
            
        diagnostics = continuity_diagnostics(values, plan.rank)
        
        if plan.state is CalibrationReadinessState.NOT_READY:
            return CalibrationReadiness(plan=plan, threshold=None, diagnostics=diagnostics)
            
        ordered = np.sort(values, kind="stable")
        threshold = float(ordered[plan.rank - 1])
        return CalibrationReadiness(plan=plan, threshold=threshold, diagnostics=diagnostics)


def continuity_diagnostics(
    scores: np.ndarray, selected_rank: PositiveCount
) -> ContinuityDiagnostics:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Continuity diagnostics require a non-empty score vector")
    if not 1 <= selected_rank <= values.size:
        raise ValueError("selected_rank must lie inside the score vector")
        
    ordered = np.sort(values, kind="stable")
    selected = float(ordered[selected_rank - 1])
    unique = np.unique(ordered)
    positive_spacing = np.diff(unique)
    
    return ContinuityDiagnostics(
        unique_score_fraction=float(unique.size / values.size),
        duplicate_count=int(values.size - unique.size),
        selected_threshold_multiplicity=int(np.count_nonzero(ordered == selected)),
        minimum_positive_spacing=None if positive_spacing.size == 0 else float(np.min(positive_spacing)),
    )


def familywise_readiness_assurance(
    client_count: PositiveCount, familywise_alpha: Probability
) -> Assurance:
    if client_count <= 0:
        raise ValueError("client_count must be positive")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must be in (0,1)")
    return 1.0 - familywise_alpha / client_count


def clopper_pearson_interval(
    counts: BinomialCounts, confidence: ConfidenceLevel
) -> ConfidenceInterval:
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


def minimum_bidirectional_sample_count(
    lower_band: Fpr, confidence: ConfidenceLevel
) -> SampleCount | None:
    if not 0.0 <= lower_band < 1.0:
        raise ValueError("lower_band must be in [0, 1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if lower_band == 0.0:
        return None
        
    tail = (1.0 - confidence) / 2.0
    estimate = max(1, math.ceil(math.log(tail) / math.log(1.0 - lower_band)))
    
    while 1.0 - tail ** (1.0 / estimate) >= lower_band:
        estimate += 1
    while estimate > 1 and 1.0 - tail ** (1.0 / (estimate - 1)) < lower_band:
        estimate -= 1
    return estimate


class ReferenceMismatchEvaluator:
    def evaluate(
        self,
        scores: np.ndarray,
        reference_threshold: Threshold,
        band: OperatingBand,
        confidence: ConfidenceLevel,
    ) -> MismatchEvidence:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Mismatch evidence requires a non-empty one-dimensional array")
        if not np.isfinite(values).all() or not math.isfinite(reference_threshold):
            raise ValueError("Mismatch scores and threshold must be finite")

        counts = BinomialCounts(
            x=int(np.count_nonzero(values > reference_threshold)), n=values.size
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
            p_low=None if high_side_only else float(binom.cdf(counts.x, counts.n, band.lower)),
            p_high=float(binom.sf(counts.x - 1, counts.n, band.upper)),
            high_side_only=high_side_only,
        )


class FleetMismatchDecision(BaseModel):
    model_config = Frozen

    client_id: ClientId
    outcome: MismatchOutcome
    low_p_value: PValue | None = None
    high_p_value: PValue


class DirectionalHypothesis(BaseModel):
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
    if not counts_by_client:
        return ()
        
    confidence = 1.0 - familywise_alpha / len(counts_by_client)
    decisions = []
    
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
    ordered = sorted(
        hypotheses,
        key=lambda item: (item.p_value, item.client_id, item.outcome),
    )
    total = len(ordered)
    rejected = set()
    
    for index, hypothesis in enumerate(ordered):
        if hypothesis.p_value <= alpha / (total - index):
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
    hypotheses = []
    diagnostics = {}
    
    for client_id in sorted(counts_by_client):
        counts = counts_by_client[client_id]
        low, high = _directional_p_values(counts, band)
        diagnostics[client_id] = (low, high)
        
        if low is not None:
            hypotheses.append(DirectionalHypothesis(client_id=client_id, outcome=MismatchOutcome.LOW, p_value=low))
        hypotheses.append(DirectionalHypothesis(client_id=client_id, outcome=MismatchOutcome.HIGH, p_value=high))

    rejected = _holm_rejected(hypotheses, familywise_alpha)
    decisions = []
    
    for client_id in sorted(counts_by_client):
        low_rejected = (client_id, MismatchOutcome.LOW) in rejected
        high_rejected = (client_id, MismatchOutcome.HIGH) in rejected
        
        if low_rejected and high_rejected:
            raise RuntimeError(f"DIRECTION_CONTRADICTION: both directions rejected for {client_id}")
            
        outcome = (
            MismatchOutcome.LOW if low_rejected
            else MismatchOutcome.HIGH if high_rejected
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
    def decide(
        self,
        reference: ReferenceThreshold,
        readiness: CalibrationReadiness,
        mismatch: MismatchEvidence,
        reject_calibration_ties: bool,
    ) -> ThresholdDecision:
        tie_count = readiness.tie_count

        if mismatch.outcome is MismatchOutcome.INSUFFICIENT_EVIDENCE:
            state, reason = DecisionState.MISMATCH_EVIDENCE_INSUFFICIENT, DecisionReason.INSUFFICIENT_MISMATCH_EVIDENCE
        elif mismatch.outcome is MismatchOutcome.NO_MATERIAL_DIFFERENCE:
            state, reason = DecisionState.REFERENCE_RETAINED, DecisionReason.NO_MATERIAL_DIFFERENCE
        elif readiness.plan.state is CalibrationReadinessState.NOT_READY or readiness.threshold is None:
            state, reason = DecisionState.CALIBRATION_DEFICIT, DecisionReason.CALIBRATION_NOT_READY
        elif reject_calibration_ties and tie_count > 1:
            state, reason = DecisionState.ASSUMPTION_VIOLATION, DecisionReason.CALIBRATION_TIE
        else:
            return ThresholdDecision(
                state=DecisionState.PERSONALIZED,
                threshold=readiness.threshold,
                source=ThresholdSource.LOCAL_CALIBRATION,
                reason=DecisionReason.LOCAL_PERSONALIZATION_ADMITTED,
                tie_count=tie_count,
            )

        return ThresholdDecision(
            state=state,
            threshold=reference.value,
            source=ThresholdSource.REFERENCE,
            reason=reason,
            tie_count=tie_count,
        )