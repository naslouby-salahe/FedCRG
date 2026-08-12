"""Regression coverage for defects found in the pre-refactor implementation."""

from fedcrg.core.enums import CalibrationReadinessState, DecisionState, MismatchOutcome
from fedcrg.core.types import OperatingBand
from fedcrg.protocol.decision import ThresholdDecisionEngine
from fedcrg.protocol.results import CalibrationReadiness, MismatchEvidence, ReadinessPlan, ReferenceThreshold


def _reference() -> ReferenceThreshold:
    return ReferenceThreshold(1.0, 10, 10, 1, 10)


def _readiness() -> CalibrationReadiness:
    return CalibrationReadiness(ReadinessPlan(100, 99, 0.99, CalibrationReadinessState.READY, OperatingBand(0.005, 0.015), 0.95), 2.0, 1)


def _mismatch(outcome: MismatchOutcome) -> MismatchEvidence:
    return MismatchEvidence(1000, 20, 0.02, None, outcome, 736, None, None)


def test_previous_string_none_bug_cannot_recur() -> None:
    decision = ThresholdDecisionEngine().decide(_reference(), _readiness(), _mismatch(MismatchOutcome.NO_MATERIAL_DIFFERENCE))
    assert decision.state is DecisionState.REFERENCE_RETAINED


def test_previous_insufficient_evidence_fallthrough_cannot_recur() -> None:
    decision = ThresholdDecisionEngine().decide(_reference(), _readiness(), _mismatch(MismatchOutcome.INSUFFICIENT_EVIDENCE))
    assert decision.state is DecisionState.MISMATCH_EVIDENCE_INSUFFICIENT
