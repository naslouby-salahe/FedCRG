"""Regression coverage for defects found in the pre-refactor implementation."""

from fedcrg.core.enums import CalibrationReadinessState, DecisionState, MismatchOutcome
from fedcrg.core.types import ConfidenceInterval, OperatingBand
from fedcrg.protocol.decision import ThresholdDecisionEngine
from fedcrg.protocol.results import (
    CalibrationReadiness,
    ContinuityDiagnostics,
    MismatchEvidence,
    ReadinessPlan,
    ReferenceThreshold,
)


def _reference() -> ReferenceThreshold:
    return ReferenceThreshold(1.0, 10, 10, 1, 10)


def _readiness() -> CalibrationReadiness:
    diagnostics = ContinuityDiagnostics(
        unique_score_fraction=1.0,
        duplicate_count=0,
        selected_threshold_multiplicity=1,
        minimum_positive_spacing=0.01,
    )
    return CalibrationReadiness(
        ReadinessPlan(100, 99, 0.99, CalibrationReadinessState.READY, OperatingBand(0.005, 0.015), 0.95),
        2.0,
        diagnostics,
    )


def _mismatch(outcome: MismatchOutcome) -> MismatchEvidence:
    return MismatchEvidence(1000, 20, 0.02, ConfidenceInterval(0.01, 0.03), outcome, 736, 0.5, 0.5)


def test_previous_string_none_bug_cannot_recur() -> None:
    decision = ThresholdDecisionEngine().decide(_reference(), _readiness(), _mismatch(MismatchOutcome.NO_MATERIAL_DIFFERENCE))
    assert decision.state is DecisionState.REFERENCE_RETAINED


def test_previous_insufficient_evidence_fallthrough_cannot_recur() -> None:
    decision = ThresholdDecisionEngine().decide(_reference(), _readiness(), _mismatch(MismatchOutcome.INSUFFICIENT_EVIDENCE))
    assert decision.state is DecisionState.MISMATCH_EVIDENCE_INSUFFICIENT
