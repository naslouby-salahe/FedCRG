import numpy as np
import pytest

from fedcrg.domain.enums import (
    CalibrationReadinessState,
    DecisionReason,
    DecisionState,
    MismatchOutcome,
    PolicyId,
    ThresholdSource,
)
from fedcrg.domain.identifiers import ClientId
from fedcrg.domain.values import ConfidenceInterval
from fedcrg.method.results import (
    CalibrationReadiness,
    ClientEvaluationResult,
    ContinuityDiagnostics,
    MismatchEvidence,
    ReadinessPlan,
    ReferenceThreshold,
    ThresholdDecision,
)
from fedcrg.thresholds.evidence import (
    BenignPolicyEvidence,
    SupervisedDevelopmentEvidence,
    empirical_quantile,
)
from fedcrg.thresholds.selection import PolicyThresholdSelector
from tests._fixtures import primary_protocol, primary_statistics

_NON_ORACLE_POLICIES = tuple(policy for policy in PolicyId if policy is not PolicyId.ORACLE_TEST)


def _protocol_result(client_id: ClientId, shift: float, mismatch: bool) -> ClientEvaluationResult:
    protocol_config = primary_protocol()
    reference = ReferenceThreshold(0.75, 1, 100, 2, 50)
    diagnostics = ContinuityDiagnostics(
        unique_score_fraction=1.0,
        duplicate_count=0,
        selected_threshold_multiplicity=1,
        minimum_positive_spacing=0.001,
    )
    readiness = CalibrationReadiness(
        ReadinessPlan(
            sample_count=2000,
            rank=1982,
            coverage_probability=0.98,
            state=CalibrationReadinessState.READY,
            band=protocol_config.band,
            assurance=0.95,
        ),
        threshold=0.82 + shift,
        diagnostics=diagnostics,
    )
    evidence = MismatchEvidence(
        sample_count=3000,
        exceedance_count=60 if mismatch else 30,
        estimated_fpr=0.02 if mismatch else 0.01,
        interval=ConfidenceInterval(0.016, 0.024),
        outcome=MismatchOutcome.HIGH if mismatch else MismatchOutcome.NO_MATERIAL_DIFFERENCE,
        minimum_sample_count=736,
        p_low=1.0,
        p_high=0.01,
    )
    decision = ThresholdDecision(
        DecisionState.PERSONALIZED if mismatch else DecisionState.REFERENCE_RETAINED,
        0.82 + shift if mismatch else 0.75,
        ThresholdSource.LOCAL_CALIBRATION if mismatch else ThresholdSource.REFERENCE,
        DecisionReason.LOCAL_PERSONALIZATION_ADMITTED
        if mismatch
        else DecisionReason.NO_MATERIAL_DIFFERENCE,
        tie_count=1,
    )
    return ClientEvaluationResult(client_id, reference, readiness, evidence, decision)


def _benign(client_id: str, shift: float = 0.0, mismatch: bool = True) -> BenignPolicyEvidence:
    typed_id = ClientId(client_id)
    return BenignPolicyEvidence(
        client_id=typed_id,
        reference_scores=np.linspace(0.0, 0.7, 500),
        mismatch_scores=np.linspace(0.0, 1.0 + shift, 3000),
        calibration_scores=np.linspace(0.0, 1.0 + shift, 2000),
        evaluation=_protocol_result(typed_id, shift, mismatch),
    )


def _supervised(benign: BenignPolicyEvidence) -> SupervisedDevelopmentEvidence:
    return SupervisedDevelopmentEvidence(
        benign=benign,
        benign_guard_scores=np.linspace(0.0, 0.4, 500),
        attack_dev_scores=np.linspace(0.8, 1.2, 500),
    )


def test_selector_returns_complete_threshold_ledger() -> None:
    benign_clients = (_benign("c1"), _benign("c2", 0.05))
    supervised_clients = tuple(_supervised(client) for client in benign_clients)
    thresholds = PolicyThresholdSelector().select(
        benign_clients,
        primary_protocol(),
        primary_statistics(),
        _NON_ORACLE_POLICIES,
        supervised_clients,
    )
    observed = {(entry.policy, entry.client_id) for entry in thresholds.entries}
    expected = {
        (policy, client.client_id) for policy in _NON_ORACLE_POLICIES for client in benign_clients
    }
    assert observed == expected
    assert thresholds.shrinkage_n0 in {100, 300, 1000, 3000, 10000}


def test_mismatch_only_uses_calibration_empirical_quantile() -> None:
    client = _benign("c1")
    result = PolicyThresholdSelector().select(
        (client,), primary_protocol(), primary_statistics(), (PolicyId.MISMATCH_ONLY,)
    )
    assert result.for_client(PolicyId.MISMATCH_ONLY, client.client_id) == empirical_quantile(
        client.calibration_scores
    )


def test_no_mismatch_ablation_retains_reference() -> None:
    client = _benign("c1", mismatch=False)
    result = PolicyThresholdSelector().select(
        (client,), primary_protocol(), primary_statistics(), (PolicyId.MISMATCH_ONLY,)
    )
    assert result.for_client(PolicyId.MISMATCH_ONLY, client.client_id) == 0.75


def test_empirical_quantile_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        empirical_quantile(np.array([]))
