from __future__ import annotations

import numpy as np
import pytest
from pydantic import TypeAdapter

from fedcrg.thresholding.metrics import ClientEvaluation
from fedcrg.thresholding.readiness import ReferenceThreshold
from fedcrg.types import (
    CalibrationReadinessState,
    ClientId,
    DecisionState,
    MismatchOutcome,
)
from tests._fixtures import primary_protocol

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_CLIENT_A = _CLIENT_ID_ADAPTER.validate_python("client-a")
_CLIENT_B = _CLIENT_ID_ADAPTER.validate_python("client-b")

_HIGH_CALIBRATION = np.linspace(0.5, 1.0, 2000)
_LOW_CALIBRATION = np.linspace(0.0, 0.4, 100)
_HIGH_MISMATCH = np.linspace(0.8, 0.9, 736)
_NO_MATERIAL_MISMATCH = np.concatenate((np.full(729, 0.6), np.full(7, 0.8)))


def _reference() -> ReferenceThreshold:
    return ReferenceThreshold(
        value=0.75,
        rank=2,
        sample_count=4,
        client_count=2,
        samples_per_client=2,
    )


def test_evaluate_client_deploys_when_evidence_holds() -> None:
    protocol = primary_protocol()
    evaluator = ClientEvaluation()
    evaluator.precompute_readiness(2000, protocol)
    result = evaluator.evaluate_client(
        _CLIENT_A,
        _reference(),
        _HIGH_CALIBRATION,
        _HIGH_MISMATCH,
        protocol,
    )
    assert result.client_id == _CLIENT_A
    assert result.decision.state is DecisionState.PERSONALIZED
    assert result.decision.threshold > 0.75
    assert result.reference.value == 0.75


def test_evaluate_client_rejects_insufficient_calibration() -> None:
    protocol = primary_protocol()
    evaluator = ClientEvaluation()
    evaluator.precompute_readiness(100, protocol)
    result = evaluator.evaluate_client(
        _CLIENT_B,
        _reference(),
        _LOW_CALIBRATION,
        _HIGH_MISMATCH,
        protocol,
    )
    assert result.readiness.plan.state is CalibrationReadinessState.NOT_READY
    assert result.readiness.threshold is None
    assert result.decision.state is DecisionState.CALIBRATION_DEFICIT
    assert result.decision.source.value == "reference"


def test_evaluate_client_retains_reference_without_mismatch() -> None:
    protocol = primary_protocol()
    evaluator = ClientEvaluation()
    evaluator.precompute_readiness(2000, protocol)
    result = evaluator.evaluate_client(
        _CLIENT_A,
        _reference(),
        _HIGH_CALIBRATION,
        _NO_MATERIAL_MISMATCH,
        protocol,
    )
    assert result.mismatch.outcome is MismatchOutcome.NO_MATERIAL_DIFFERENCE
    assert result.decision.state is DecisionState.REFERENCE_RETAINED
    assert result.decision.threshold == 0.75
    assert result.decision.source.value == "reference"


def test_evaluate_client_requires_precomputed_readiness() -> None:
    protocol = primary_protocol()
    evaluator = ClientEvaluation()
    with pytest.raises(FileNotFoundError):
        evaluator.evaluate_client(
            _CLIENT_A,
            _reference(),
            _HIGH_CALIBRATION,
            _HIGH_MISMATCH,
            protocol,
        )


def test_estimate_reference_pools_equal_counts() -> None:
    evaluator = ClientEvaluation()
    reference = evaluator.estimate_reference(
        {
            _CLIENT_A: np.zeros(3),
            _CLIENT_B: np.ones(3),
        },
        primary_protocol(),
    )
    assert reference.sample_count == 6
    assert reference.client_count == 2
    assert reference.samples_per_client == 3
    assert reference.rank == 6
