"""Deterministic threshold-deployment decision state machine."""

from fedcrg.domain.enums import (
    CalibrationReadinessState,
    DecisionReason,
    DecisionState,
    MismatchOutcome,
    ThresholdSource,
)
from fedcrg.decision.results import (
    CalibrationReadiness,
    MismatchEvidence,
    ReferenceThreshold,
    ThresholdDecision,
)


class ThresholdDecisionEngine:
    """Combine independent evidence without reinterpreting inconclusive states."""

    def decide(
        self,
        reference: ReferenceThreshold,
        readiness: CalibrationReadiness,
        mismatch: MismatchEvidence,
        reject_calibration_ties: bool = True,
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
        if (
            readiness.plan.state is CalibrationReadinessState.NOT_READY
            or readiness.threshold is None
        ):
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
