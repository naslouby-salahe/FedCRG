import numpy as np
import pytest

from fedcrg.config.models import ProtocolConfig
from fedcrg.core.enums import CalibrationReadinessState, DecisionReason, DecisionState, MismatchOutcome, PolicyId, ThresholdSource
from fedcrg.core.types import ConfidenceInterval
from fedcrg.policies.base import ClientPolicyData, empirical_quantile
from fedcrg.policies.registry import FederationPolicySelector
from fedcrg.protocol.results import CalibrationReadiness, ClientProtocolResult, MismatchEvidence, ReadinessPlan, ReferenceThreshold, ThresholdDecision


def _client(client_id: str, shift: float = 0.0, mismatch: bool = True) -> ClientPolicyData:
    protocol_config = ProtocolConfig()
    reference = ReferenceThreshold(0.75, 1, 100, 2, 50)
    readiness = CalibrationReadiness(ReadinessPlan(sample_count=2000, rank=1982, coverage_probability=0.98, state=CalibrationReadinessState.READY, band=protocol_config.band, assurance=0.95), threshold=0.82 + shift, tie_count=1)
    evidence = MismatchEvidence(sample_count=3000, exceedance_count=60 if mismatch else 30, estimated_fpr=0.02 if mismatch else 0.01, interval=ConfidenceInterval(0.016, 0.024), outcome=MismatchOutcome.HIGH if mismatch else MismatchOutcome.NO_MATERIAL_DIFFERENCE, minimum_sample_count=736, p_low=1.0, p_high=0.01)
    decision = ThresholdDecision(DecisionState.PERSONALIZED if mismatch else DecisionState.REFERENCE_RETAINED, 0.82 + shift if mismatch else 0.75, ThresholdSource.LOCAL_CALIBRATION if mismatch else ThresholdSource.REFERENCE, DecisionReason.LOCAL_PERSONALIZATION_ADMITTED if mismatch else DecisionReason.NO_MATERIAL_DIFFERENCE)
    result = ClientProtocolResult(client_id, reference, readiness, evidence, decision)
    return ClientPolicyData(client_id, np.linspace(0.0, 0.7, 500), np.linspace(0.0, 1.0 + shift, 3000), np.linspace(0.0, 1.0 + shift, 2000), np.linspace(0.0, 0.4, 500), np.linspace(0.8, 1.2, 500), np.linspace(0.0, 0.6, 1000), np.linspace(0.7, 1.3, 1000), tuple("attack-a" if i < 500 else "attack-b" for i in range(1000)), result)


def test_selector_returns_complete_threshold_ledger() -> None:
    clients = (_client("c1"), _client("c2", 0.05))
    thresholds = FederationPolicySelector().select(clients, ProtocolConfig())
    assert set(thresholds.values) == set(PolicyId)
    assert all(set(mapping) == {"c1", "c2"} for mapping in thresholds.values.values())
    assert thresholds.shrinkage_n0 in {100, 300, 1000, 3000, 10000}


def test_mismatch_only_uses_calibration_empirical_quantile() -> None:
    client = _client("c1")
    result = FederationPolicySelector().select((client,), ProtocolConfig())
    assert result.for_client(PolicyId.MISMATCH_ONLY, "c1") == empirical_quantile(client.calibration_scores)


def test_no_mismatch_ablation_retains_reference() -> None:
    client = _client("c1", mismatch=False)
    assert FederationPolicySelector().select((client,), ProtocolConfig()).for_client(PolicyId.MISMATCH_ONLY, "c1") == 0.75


def test_empirical_quantile_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        empirical_quantile(np.array([]))
