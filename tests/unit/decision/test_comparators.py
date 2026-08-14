"""Unit tests for result models and the complete threshold-selection ledger."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import TypeAdapter

from fedcrg.thresholding.policies import (
    BenignPolicyEvidence,
    PolicyThresholdSelector,
    SupervisedDevelopmentEvidence,
    empirical_quantile,
)
from fedcrg.thresholding.readiness import (
    CalibrationReadiness,
    CalibrationReadinessEvaluator,
    ClientEvaluationResult,
    MismatchEvidence,
    ReadinessPlanBuilder,
    ReferenceThreshold,
    ThresholdDecision,
)
from fedcrg.types import (
    ClientId,
    ConfidenceInterval,
    DecisionReason,
    DecisionState,
    MismatchOutcome,
    OperatingBand,
    PolicyId,
    ThresholdSource,
)
from tests._fixtures import primary_protocol, primary_statistics

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_CLIENT_A = _CLIENT_ID_ADAPTER.validate_python("client-a")
_CLIENT_B = _CLIENT_ID_ADAPTER.validate_python("client-b")

_NON_ORACLE_POLICIES = tuple(policy for policy in PolicyId if policy is not PolicyId.ORACLE_TEST)


def _reference() -> ReferenceThreshold:
    return ReferenceThreshold(
        value=0.75,
        rank=2,
        sample_count=4,
        client_count=2,
        samples_per_client=2,
    )


def _readiness() -> CalibrationReadiness:
    band = OperatingBand(lower=0.005, upper=0.015)
    plan = ReadinessPlanBuilder().build(2000, band, 0.95)
    evaluator = CalibrationReadinessEvaluator()
    result = evaluator.evaluate(np.linspace(0.5, 1.0, 2000), plan)
    assert result.tie_count == 1
    return result


def _mismatch(
    sample_count: int = 500,
    exceedance_count: int = 0,
    outcome: MismatchOutcome = MismatchOutcome.NO_MATERIAL_DIFFERENCE,
) -> MismatchEvidence:
    return MismatchEvidence(
        sample_count=sample_count,
        exceedance_count=exceedance_count,
        estimated_fpr=0.0 if exceedance_count == 0 else 0.02,
        interval=ConfidenceInterval(lower=0.0, upper=0.0),
        outcome=outcome,
        minimum_sample_count=736,
        p_low=0.5,
        p_high=0.5,
    )


def _threshold_decision(state: DecisionState = DecisionState.PERSONALIZED) -> ThresholdDecision:
    return ThresholdDecision(
        state=state,
        threshold=0.82,
        source=ThresholdSource.LOCAL_CALIBRATION,
        reason=DecisionReason.LOCAL_PERSONALIZATION_ADMITTED,
        tie_count=0,
    )


def _supervised() -> tuple[SupervisedDevelopmentEvidence, ...]:
    benign_evidence = _benign_clients()
    benign = np.linspace(0.0, 0.4, 500)
    attack = np.linspace(0.8, 1.2, 500)
    return tuple(
        SupervisedDevelopmentEvidence(
            benign=benign_evidence[index],
            benign_guard_scores=benign,
            attack_dev_scores=attack,
        )
        for index in range(2)
    )


def _benign_clients() -> tuple[BenignPolicyEvidence, ...]:
    readiness = _readiness()
    clients = []
    for client in (_CLIENT_A, _CLIENT_B):
        reference = _reference()
        calibration = np.linspace(0.5, 1.0, 2000)
        mismatch_scores = np.linspace(0.8, 0.9, 736)
        mismatch = _mismatch(
            sample_count=736,
            exceedance_count=736,
            outcome=MismatchOutcome.HIGH,
        )
        decision = _threshold_decision()
        clients.append(
            BenignPolicyEvidence(
                client_id=client,
                reference_scores=np.linspace(0.4, 0.9, 10),
                mismatch_scores=mismatch_scores,
                calibration_scores=calibration,
                evaluation=ClientEvaluationResult(
                    client_id=client,
                    reference=reference,
                    readiness=readiness,
                    mismatch=mismatch,
                    decision=decision,
                ),
            )
        )
    return tuple(clients)


def test_result_models_serialize_and_roundtrip() -> None:
    reference = _reference()
    readiness = _readiness()
    mismatch = _mismatch()
    decision = _threshold_decision()
    result = ClientEvaluationResult(
        client_id=_CLIENT_A,
        reference=reference,
        readiness=readiness,
        mismatch=mismatch,
        decision=decision,
    )
    decoded = ClientEvaluationResult.model_validate_json(result.model_dump_json())
    assert decoded == result
    assert decoded.decision.state is DecisionState.PERSONALIZED
    assert decoded.decision.threshold == 0.82


def test_selector_returns_complete_threshold_ledger() -> None:
    protocol = primary_protocol()
    statistics = primary_statistics()
    thresholds = PolicyThresholdSelector().select(
        _benign_clients(),
        protocol,
        statistics,
        _NON_ORACLE_POLICIES,
        _supervised(),
    )
    expected_pairs = {
        (policy, client) for policy in _NON_ORACLE_POLICIES for client in (_CLIENT_A, _CLIENT_B)
    }
    actual_pairs = {(entry.policy, entry.client_id) for entry in thresholds.entries}
    assert actual_pairs == expected_pairs
    assert thresholds.shrinkage_n0 in statistics.shrinkage_n0_candidates


def test_selector_threshold_matches_local_quantile_for_mismatch_only() -> None:
    protocol = primary_protocol()
    thresholds = PolicyThresholdSelector().select(
        _benign_clients(),
        protocol,
        primary_statistics(),
        (PolicyId.MISMATCH_ONLY,),
    )
    for client in (_CLIENT_A, _CLIENT_B):
        expected = empirical_quantile(np.linspace(0.5, 1.0, 2000), protocol.alpha)
        assert thresholds.for_client(PolicyId.MISMATCH_ONLY, client) == expected


def test_selector_for_client_raises_for_oracle_before_final_evidence() -> None:
    thresholds = PolicyThresholdSelector().select(
        _benign_clients(),
        primary_protocol(),
        primary_statistics(),
        (PolicyId.FEDCRG,),
    )
    fedcrg = thresholds.for_client(PolicyId.FEDCRG, _CLIENT_A)
    assert fedcrg is not None
    assert fedcrg == pytest.approx(0.82)
    with pytest.raises(ValueError):
        thresholds.for_client(PolicyId.ORACLE_TEST, _CLIENT_A)


def test_selector_uses_decision_threshold_for_fedcrg() -> None:
    protocol = primary_protocol()
    thresholds = PolicyThresholdSelector().select(
        _benign_clients(),
        protocol,
        primary_statistics(),
        (PolicyId.FEDCRG,),
    )
    for client in (_CLIENT_A, _CLIENT_B):
        assert thresholds.for_client(PolicyId.FEDCRG, client) == pytest.approx(0.82)
