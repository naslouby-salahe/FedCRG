from fedcrg.core.enums import CalibrationReadinessState, DecisionState, MismatchOutcome, ThresholdSource
from fedcrg.core.types import OperatingBand
from fedcrg.protocol.decision import ThresholdDecisionEngine
from fedcrg.protocol.results import CalibrationReadiness, MismatchEvidence, ReadinessPlan, ReferenceThreshold


def _reference() -> ReferenceThreshold:
    return ReferenceThreshold(1.0, 10, 10, 1, 10)


def _readiness(ready: bool = True, ties: int = 1) -> CalibrationReadiness:
    plan = ReadinessPlan(sample_count=100, rank=99, coverage_probability=0.99 if ready else 0.5, state=CalibrationReadinessState.READY if ready else CalibrationReadinessState.NOT_READY, band=OperatingBand(0.005, 0.015), assurance=0.95)
    return CalibrationReadiness(plan, 2.0 if ready else None, ties)


def _mismatch(outcome: MismatchOutcome) -> MismatchEvidence:
    return MismatchEvidence(1000, 20, 0.02, None, outcome, 736, None, None)


def test_no_material_difference_always_retains_reference() -> None:
    decision = ThresholdDecisionEngine().decide(_reference(), _readiness(), _mismatch(MismatchOutcome.NO_MATERIAL_DIFFERENCE))
    assert decision.state is DecisionState.REFERENCE_RETAINED
    assert decision.source is ThresholdSource.REFERENCE
    assert decision.threshold == 1.0


def test_insufficient_evidence_never_personalizes() -> None:
    decision = ThresholdDecisionEngine().decide(_reference(), _readiness(), _mismatch(MismatchOutcome.INSUFFICIENT_EVIDENCE))
    assert decision.state is DecisionState.MISMATCH_EVIDENCE_INSUFFICIENT
    assert decision.source is ThresholdSource.REFERENCE


def test_personalization_requires_difference_and_ready_calibration() -> None:
    assert ThresholdDecisionEngine().decide(_reference(), _readiness(), _mismatch(MismatchOutcome.HIGH)).state is DecisionState.PERSONALIZED


def test_not_ready_calibration_retains_reference() -> None:
    assert ThresholdDecisionEngine().decide(_reference(), _readiness(ready=False), _mismatch(MismatchOutcome.HIGH)).state is DecisionState.CALIBRATION_DEFICIT


def test_calibration_tie_blocks_personalization() -> None:
    assert ThresholdDecisionEngine().decide(_reference(), _readiness(ties=2), _mismatch(MismatchOutcome.HIGH)).state is DecisionState.ASSUMPTION_VIOLATION
