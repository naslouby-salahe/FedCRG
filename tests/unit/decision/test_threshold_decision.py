"""Unit tests for the deployment decision engine."""

from __future__ import annotations

import numpy as np

from fedcrg.thresholding.metrics import ClientEvaluation
from fedcrg.thresholding.readiness import (
    CalibrationReadinessEvaluator,
    DeploymentDecision,
    ReadinessPlan,
    ReadinessPlanBuilder,
    ReferenceThreshold,
    ThresholdDecision,
)
from fedcrg.types import DecisionState, OperatingBand
from tests._fixtures import primary_protocol

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


def _evaluate(calibration_scores, mismatch_scores, plan: ReadinessPlan) -> ThresholdDecision:
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


def test_decide_personalizes_when_evidence_holds() -> None:
    decision = _evaluate(_HIGH_CALIBRATION, _HIGH_MISMATCH, _PLAN)
    assert decision.state is DecisionState.PERSONALIZED
    assert decision.threshold > _REFERENCE.value
    assert decision.source.value == "local_calibration"


def test_decide_retains_reference_when_no_material_mismatch() -> None:
    decision = _evaluate(_HIGH_CALIBRATION, _NO_MATERIAL_MISMATCH, _PLAN)
    assert decision.state is DecisionState.REFERENCE_RETAINED
    assert decision.threshold == _REFERENCE.value
    assert decision.source.value == "reference"


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
