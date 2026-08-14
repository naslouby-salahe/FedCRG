"""Regression tests for locked decision semantics."""

from __future__ import annotations

from pydantic import TypeAdapter

from fedcrg.thresholding.readiness import (
    CalibrationReadiness,
    ContinuityDiagnostics,
    DeploymentDecision,
    MismatchEvidence,
    ReadinessPlanBuilder,
    ReferenceThreshold,
    ThresholdDecision,
)
from fedcrg.types import (
    CalibrationReadinessState,
    ClientId,
    ConfidenceInterval,
    DecisionState,
    MismatchOutcome,
    OperatingBand,
    ThresholdSource,
)

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_CLIENT = _CLIENT_ID_ADAPTER.validate_python("client-a")

_BAND = OperatingBand(lower=0.005, upper=0.015)
_READY_PLAN = ReadinessPlanBuilder().build(2000, _BAND, 0.95)


def _reference() -> ReferenceThreshold:
    return ReferenceThreshold(
        value=1.0,
        rank=10,
        sample_count=10,
        client_count=1,
        samples_per_client=10,
    )


def _mismatch(outcome: MismatchOutcome) -> MismatchEvidence:
    return MismatchEvidence(
        sample_count=1000,
        exceedance_count=20,
        estimated_fpr=0.02,
        interval=ConfidenceInterval(lower=0.01, upper=0.03),
        outcome=outcome,
        minimum_sample_count=736,
        p_low=0.5,
        p_high=0.5,
    )


def _readiness(threshold: float | None, state: CalibrationReadinessState) -> CalibrationReadiness:
    plan = (
        ReadinessPlanBuilder().build(100, _BAND, 0.95)
        if state is CalibrationReadinessState.NOT_READY
        else _READY_PLAN
    )
    diagnostics = ContinuityDiagnostics(
        unique_score_fraction=1.0,
        duplicate_count=0,
        selected_threshold_multiplicity=1,
        minimum_positive_spacing=0.001,
    )
    return CalibrationReadiness(plan=plan, threshold=threshold, diagnostics=diagnostics)


def test_reference_retained_when_no_mismatch() -> None:
    decision = DeploymentDecision().decide(
        reference=_reference(),
        readiness=_readiness(2.0, CalibrationReadinessState.READY),
        mismatch=_mismatch(MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        reject_calibration_ties=True,
    )
    assert decision.state is DecisionState.REFERENCE_RETAINED
    assert decision.threshold == 1.0
    assert decision.source is ThresholdSource.REFERENCE


def test_mismatch_evidence_insufficient_blocks_personalization() -> None:
    decision = DeploymentDecision().decide(
        reference=_reference(),
        readiness=_readiness(2.0, CalibrationReadinessState.READY),
        mismatch=_mismatch(MismatchOutcome.INSUFFICIENT_EVIDENCE),
        reject_calibration_ties=True,
    )
    assert decision.state is DecisionState.MISMATCH_EVIDENCE_INSUFFICIENT
    assert decision.threshold == 1.0


def test_calibration_deficit_keeps_reference_threshold() -> None:
    decision = DeploymentDecision().decide(
        reference=_reference(),
        readiness=_readiness(None, CalibrationReadinessState.NOT_READY),
        mismatch=_mismatch(MismatchOutcome.HIGH),
        reject_calibration_ties=True,
    )
    assert decision.state is DecisionState.CALIBRATION_DEFICIT
    assert decision.threshold == 1.0
    assert decision.source is ThresholdSource.REFERENCE


def test_personalization_uses_local_threshold() -> None:
    decision = DeploymentDecision().decide(
        reference=_reference(),
        readiness=_readiness(2.0, CalibrationReadinessState.READY),
        mismatch=_mismatch(MismatchOutcome.HIGH),
        reject_calibration_ties=True,
    )
    assert decision.state is DecisionState.PERSONALIZED
    assert decision.threshold == 2.0
    assert decision.source is ThresholdSource.LOCAL_CALIBRATION


def test_threshold_decision_serialization_roundtrip() -> None:
    decision = DeploymentDecision().decide(
        reference=_reference(),
        readiness=_readiness(2.0, CalibrationReadinessState.READY),
        mismatch=_mismatch(MismatchOutcome.HIGH),
        reject_calibration_ties=True,
    )
    decoded = ThresholdDecision.model_validate_json(decision.model_dump_json())
    assert decoded == decision
