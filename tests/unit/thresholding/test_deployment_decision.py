from __future__ import annotations

import numpy as np

from fedcrg.thresholding.metrics import ClientEvaluation
from fedcrg.thresholding.readiness import (
    CalibrationReadiness,
    CalibrationReadinessEvaluator,
    ContinuityDiagnostics,
    DeploymentDecision,
    MismatchEvidence,
    ReadinessPlan,
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
from pydantic import TypeAdapter

from tests._fixtures import primary_protocol

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_CLIENT = _CLIENT_ID_ADAPTER.validate_python("client-a")

_BAND = OperatingBand(lower=0.005, upper=0.015)
_PLAN = ReadinessPlanBuilder().build(2000, _BAND, 0.95)
_SMALL_PLAN = ReadinessPlanBuilder().build(100, _BAND, 0.95)
_REFERENCE = ReferenceThreshold(
    value=0.75,
    rank=2,
    sample_count=4,
    client_count=2,
    samples_per_client=2,
)

_HIGH_CALIBRATION = np.linspace(0.5, 1.0, 2000)
_HIGH_MISMATCH = np.linspace(0.8, 0.9, 736)
_LOW_MISMATCH = np.linspace(0.1, 0.2, 736)
_NO_MATERIAL_MISMATCH = np.concatenate((np.full(729, 0.6), np.full(7, 0.8)))


def _evaluate(
    calibration_scores: np.ndarray,
    mismatch_scores: np.ndarray,
    plan: ReadinessPlan,
) -> ThresholdDecision:
    protocol = primary_protocol()
    evaluator = CalibrationReadinessEvaluator()
    readiness = evaluator.evaluate(np.asarray(calibration_scores, dtype=np.float64), plan)
    mismatch = ClientEvaluation().mismatch_evaluator.evaluate(
        scores=np.asarray(mismatch_scores, dtype=np.float64),
        reference_threshold=_REFERENCE.value,
        band=protocol.band,
        confidence=protocol.mismatch_confidence,
    )
    return DeploymentDecision().decide(
        reference=_REFERENCE,
        readiness=readiness,
        mismatch=mismatch,
        reject_calibration_ties=protocol.reject_calibration_ties,
    )


def _hand_built_mismatch(outcome: MismatchOutcome) -> MismatchEvidence:
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


def _hand_built_readiness(
    threshold: float | None, state: CalibrationReadinessState
) -> CalibrationReadiness:
    plan = (
        ReadinessPlanBuilder().build(100, _BAND, 0.95)
        if state is CalibrationReadinessState.NOT_READY
        else _PLAN
    )
    diagnostics = ContinuityDiagnostics(
        unique_score_fraction=1.0,
        duplicate_count=0,
        selected_threshold_multiplicity=1,
        minimum_positive_spacing=0.001,
    )
    return CalibrationReadiness(plan=plan, threshold=threshold, diagnostics=diagnostics)


def _hand_built_reference() -> ReferenceThreshold:
    return ReferenceThreshold(
        value=1.0,
        rank=10,
        sample_count=10,
        client_count=1,
        samples_per_client=10,
    )


def test_decide_personalizes_when_evidence_holds() -> None:
    decision = _evaluate(_HIGH_CALIBRATION, _HIGH_MISMATCH, _PLAN)
    assert decision.state is DecisionState.PERSONALIZED
    assert decision.threshold > _REFERENCE.value
    assert decision.source == "local_calibration"


def test_decide_retains_reference_when_no_material_mismatch() -> None:
    decision = _evaluate(_HIGH_CALIBRATION, _NO_MATERIAL_MISMATCH, _PLAN)
    assert decision.state is DecisionState.REFERENCE_RETAINED
    assert decision.threshold == _REFERENCE.value
    assert decision.source == "reference"


def test_decide_refuses_without_calibration_readiness() -> None:
    decision = _evaluate(np.linspace(0.0, 0.4, 100), _HIGH_MISMATCH, _SMALL_PLAN)
    assert decision.state is DecisionState.CALIBRATION_DEFICIT
    assert decision.threshold == _REFERENCE.value


def test_decide_honors_reject_calibration_ties_flag() -> None:
    protocol = primary_protocol()
    evaluator = CalibrationReadinessEvaluator()
    readiness = evaluator.evaluate(_HIGH_CALIBRATION, _PLAN)
    mismatch = ClientEvaluation().mismatch_evaluator.evaluate(
        scores=_HIGH_MISMATCH,
        reference_threshold=_REFERENCE.value,
        band=protocol.band,
        confidence=protocol.mismatch_confidence,
    )
    accepted = DeploymentDecision().decide(
        reference=_REFERENCE,
        readiness=readiness,
        mismatch=mismatch,
        reject_calibration_ties=False,
    )
    assert accepted.state is not DecisionState.ASSUMPTION_VIOLATION


def test_decide_is_insensitive_to_mismatch_direction() -> None:
    high = _evaluate(_HIGH_CALIBRATION, _HIGH_MISMATCH, _PLAN)
    low = _evaluate(_HIGH_CALIBRATION, _LOW_MISMATCH, _PLAN)
    assert high.state is DecisionState.PERSONALIZED
    assert low.state is DecisionState.PERSONALIZED


def test_decide_flags_assumption_violation_on_calibration_tie() -> None:
    """A duplicate value at the selected rank makes the order statistic ambiguous, so ties
    block personalization by default rather than being silently jittered away."""
    plan = ReadinessPlanBuilder().build(2000, _BAND, 0.95)
    values = np.linspace(0.5, 1.0, 2000)
    values[plan.rank - 8 : plan.rank + 7] = values[plan.rank - 1]
    readiness = CalibrationReadinessEvaluator().evaluate(values, plan)
    assert readiness.plan.state is CalibrationReadinessState.READY
    assert readiness.tie_count > 1

    reference = ReferenceThreshold(
        value=0.75, rank=2, sample_count=4, client_count=2, samples_per_client=2
    )
    mismatch = ClientEvaluation().mismatch_evaluator.evaluate(
        scores=np.linspace(0.8, 0.9, 736),
        reference_threshold=reference.value,
        band=_BAND,
        confidence=primary_protocol().mismatch_confidence,
    )
    assert mismatch.outcome is MismatchOutcome.HIGH

    decision = DeploymentDecision().decide(
        reference=reference,
        readiness=readiness,
        mismatch=mismatch,
        reject_calibration_ties=True,
    )
    assert decision.state.value == "CALIBRATION_ASSUMPTION_VIOLATION"
    assert decision.threshold == reference.value
    assert decision.tie_count == readiness.tie_count

    admitted = DeploymentDecision().decide(
        reference=reference,
        readiness=readiness,
        mismatch=mismatch,
        reject_calibration_ties=False,
    )
    assert admitted.state.value == "LOCAL_PERSONALIZE"
    assert admitted.threshold == readiness.threshold


def test_reference_retained_when_no_mismatch() -> None:
    decision = DeploymentDecision().decide(
        reference=_hand_built_reference(),
        readiness=_hand_built_readiness(2.0, CalibrationReadinessState.READY),
        mismatch=_hand_built_mismatch(MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        reject_calibration_ties=True,
    )
    assert decision.state is DecisionState.REFERENCE_RETAINED
    assert decision.threshold == 1.0
    assert decision.source is ThresholdSource.REFERENCE


def test_mismatch_evidence_insufficient_blocks_personalization() -> None:
    decision = DeploymentDecision().decide(
        reference=_hand_built_reference(),
        readiness=_hand_built_readiness(2.0, CalibrationReadinessState.READY),
        mismatch=_hand_built_mismatch(MismatchOutcome.INSUFFICIENT_EVIDENCE),
        reject_calibration_ties=True,
    )
    assert decision.state is DecisionState.MISMATCH_EVIDENCE_INSUFFICIENT
    assert decision.threshold == 1.0


def test_calibration_deficit_keeps_reference_threshold() -> None:
    decision = DeploymentDecision().decide(
        reference=_hand_built_reference(),
        readiness=_hand_built_readiness(None, CalibrationReadinessState.NOT_READY),
        mismatch=_hand_built_mismatch(MismatchOutcome.HIGH),
        reject_calibration_ties=True,
    )
    assert decision.state is DecisionState.CALIBRATION_DEFICIT
    assert decision.threshold == 1.0
    assert decision.source is ThresholdSource.REFERENCE


def test_personalization_uses_local_threshold() -> None:
    decision = DeploymentDecision().decide(
        reference=_hand_built_reference(),
        readiness=_hand_built_readiness(2.0, CalibrationReadinessState.READY),
        mismatch=_hand_built_mismatch(MismatchOutcome.HIGH),
        reject_calibration_ties=True,
    )
    assert decision.state is DecisionState.PERSONALIZED
    assert decision.threshold == 2.0
    assert decision.source is ThresholdSource.LOCAL_CALIBRATION


def test_threshold_decision_serialization_roundtrip() -> None:
    decision = DeploymentDecision().decide(
        reference=_hand_built_reference(),
        readiness=_hand_built_readiness(2.0, CalibrationReadinessState.READY),
        mismatch=_hand_built_mismatch(MismatchOutcome.HIGH),
        reject_calibration_ties=True,
    )
    decoded = ThresholdDecision.model_validate_json(decision.model_dump_json())
    assert decoded == decision
