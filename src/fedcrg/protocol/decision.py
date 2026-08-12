"""The only threshold-decision state machine in the codebase."""

from fedcrg.core.enums import (
    CalibrationReadinessState,
    DecisionReason,
    DecisionState,
    MismatchOutcome,
    ThresholdSource,
)
from fedcrg.protocol.results import (
    CalibrationReadiness,
    MismatchEvidence,
    ReferenceThreshold,
    ThresholdDecision,
)


class ThresholdDecisionEngine:
    def decide(
        self,
        reference: ReferenceThreshold,
        readiness: CalibrationReadiness,
        mismatch: MismatchEvidence,
        reject_calibration_ties: bool = True,
    ) -> ThresholdDecision:
        if mismatch.outcome is MismatchOutcome.INSUFFICIENT_EVIDENCE:
            return ThresholdDecision(
                state=DecisionState.MISMATCH_EVIDENCE_INSUFFICIENT,
                threshold=reference.value,
                source=ThresholdSource.REFERENCE,
                reason=DecisionReason.INSUFFICIENT_MISMATCH_EVIDENCE,
            )
        if mismatch.outcome is MismatchOutcome.NO_MATERIAL_DIFFERENCE:
            return ThresholdDecision(
                state=DecisionState.REFERENCE_RETAINED,
                threshold=reference.value,
                source=ThresholdSource.REFERENCE,
                reason=DecisionReason.NO_MATERIAL_DIFFERENCE,
            )
        if readiness.plan.state is CalibrationReadinessState.NOT_READY or readiness.threshold is None:
            return ThresholdDecision(
                state=DecisionState.CALIBRATION_DEFICIT,
                threshold=reference.value,
                source=ThresholdSource.REFERENCE,
                reason=DecisionReason.CALIBRATION_NOT_READY,
            )
        if reject_calibration_ties and readiness.tie_count > 1:
            return ThresholdDecision(
                state=DecisionState.ASSUMPTION_VIOLATION,
                threshold=reference.value,
                source=ThresholdSource.REFERENCE,
                reason=DecisionReason.CALIBRATION_TIE,
            )
        return ThresholdDecision(
            state=DecisionState.PERSONALIZED,
            threshold=readiness.threshold,
            source=ThresholdSource.LOCAL_CALIBRATION,
            reason=DecisionReason.LOCAL_PERSONALIZATION_ADMITTED,
        )
