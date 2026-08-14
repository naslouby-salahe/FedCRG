"""Unit tests for policy selection: information regime and deployability."""

from __future__ import annotations

from pydantic import TypeAdapter

from fedcrg.thresholding.policies import (
    SUPERVISED_POLICIES,
    information_regime,
    is_deployable,
)
from fedcrg.types import ClientId, InformationRegime, PolicyId

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_CLIENT = _CLIENT_ID_ADAPTER.validate_python("client-a")


def test_information_regime_is_locked() -> None:
    assert information_regime(PolicyId.FEDCRG) is InformationRegime.BENIGN_ONLY
    assert information_regime(PolicyId.GLOBAL_QUANTILE) is InformationRegime.BENIGN_ONLY
    assert information_regime(PolicyId.LOCAL_QUANTILE) is InformationRegime.BENIGN_ONLY
    assert information_regime(PolicyId.READINESS_ONLY) is InformationRegime.BENIGN_ONLY
    assert information_regime(PolicyId.MISMATCH_ONLY) is InformationRegime.BENIGN_ONLY
    assert information_regime(PolicyId.SHRINKAGE) is InformationRegime.BENIGN_ONLY
    assert information_regime(PolicyId.THREE_SIGMA) is InformationRegime.BENIGN_ONLY
    assert information_regime(PolicyId.ORACLE_TEST) is InformationRegime.FINAL_TEST_ORACLE
    assert information_regime(PolicyId.DEV_F1_SELECT) is InformationRegime.SUPERVISED_DEVELOPMENT
    assert (
        information_regime(PolicyId.SUMMARY_STATISTIC_SELECT)
        is InformationRegime.SUPERVISED_DEVELOPMENT
    )
    assert information_regime(PolicyId.SUPERVISED_F1) is InformationRegime.SUPERVISED_DEVELOPMENT


def test_supervised_policies_are_locked() -> None:
    assert SUPERVISED_POLICIES == frozenset(
        {
            PolicyId.DEV_F1_SELECT,
            PolicyId.SUMMARY_STATISTIC_SELECT,
            PolicyId.SUPERVISED_F1,
        }
    )


def test_is_deployable_matches_regime() -> None:
    for policy in PolicyId:
        expected = policy not in SUPERVISED_POLICIES and policy is not PolicyId.ORACLE_TEST
        assert is_deployable(policy) is expected
