"""Contract: the central policy registry is complete and internally consistent.

Every PolicyId member must have exactly one PolicySpec; the registry must be
the single source for the information regime, deployability, supervised
status, required evidence bundles, threshold origin, and ledger payload of
every policy. The study configuration's policy list must match the registry.
"""

from __future__ import annotations

from fedcrg.config import Study
from fedcrg.thresholding.policies import (
    POLICIES,
    PolicyStrategy,
    PolicyUploadKind,
    ThresholdOrigin,
    information_regime,
    is_deployable,
)
from fedcrg.types import InformationRegime, PolicyId


def test_registry_contains_every_policy_exactly_once() -> None:
    assert set(POLICIES) == set(PolicyId)
    for policy, spec in POLICIES.items():
        assert spec.policy is policy
    assert len(POLICIES) == len(PolicyId)


def test_registry_drives_regime_and_deployability_helpers() -> None:
    for policy in PolicyId:
        assert information_regime(policy) is POLICIES[policy].regime
        assert is_deployable(policy) is POLICIES[policy].deployable


def test_supervised_policies_are_never_deployable_and_use_development_origin() -> None:
    for spec in POLICIES.values():
        if spec.supervised:
            assert not spec.deployable
            assert spec.regime is InformationRegime.SUPERVISED_DEVELOPMENT
            assert spec.threshold_origin is ThresholdOrigin.DEVELOPMENT
            assert spec.requires_supervised_development
        if spec.requires_supervised_development:
            assert spec.supervised


def test_oracle_requires_final_test_evidence_and_is_not_deployable() -> None:
    oracle = POLICIES[PolicyId.ORACLE_TEST]
    assert not oracle.deployable
    assert oracle.regime is InformationRegime.FINAL_TEST_ORACLE
    assert oracle.threshold_origin is ThresholdOrigin.FINAL_TEST
    assert oracle.requires_final_test
    assert len(oracle.candidate_policies) == 3
    for candidate in oracle.candidate_policies:
        assert candidate is not PolicyId.ORACLE_TEST
        assert not POLICIES[candidate].requires_final_test


def test_deployable_policies_are_benign_only_and_need_no_final_test() -> None:
    for spec in POLICIES.values():
        if spec.deployable:
            assert not spec.supervised
            assert spec.regime is InformationRegime.BENIGN_ONLY
            assert not spec.requires_final_test
            assert spec.threshold_origin is ThresholdOrigin.CALIBRATION


def test_reference_injection_semantics_are_registry_owned() -> None:
    reference_injecting = {
        PolicyId.REFERENCE_QUANTILE,
        PolicyId.READINESS_ONLY,
        PolicyId.MISMATCH_ONLY,
        PolicyId.SHRINKAGE,
        PolicyId.FEDCRG,
    }
    for policy, spec in POLICIES.items():
        assert spec.requires_reference is (policy in reference_injecting)


def test_every_strategy_is_distinct_and_bound() -> None:
    strategies = [spec.strategy for spec in POLICIES.values()]
    assert len(set(strategies)) == len(strategies)
    for spec in POLICIES.values():
        assert isinstance(spec.strategy, PolicyStrategy)


def test_study_policy_list_matches_registry() -> None:
    study = Study.load()
    assert set(study.study_config.policies) == set(POLICIES)


def test_ledger_payload_kinds_are_closed() -> None:
    for spec in POLICIES.values():
        assert isinstance(spec.upload_payload, PolicyUploadKind)
    assert PolicyUploadKind.NONE in {spec.upload_payload for spec in POLICIES.values()}
