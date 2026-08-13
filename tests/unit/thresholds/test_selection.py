from fedcrg.domain.enums import PolicyId
from fedcrg.thresholds.selection import SUPERVISED_POLICIES, information_regime, is_deployable
from fedcrg.thresholds.results import InformationRegime


def test_information_regime_covers_every_policy() -> None:
    assert information_regime(PolicyId.ORACLE_TEST) is InformationRegime.FINAL_TEST_ORACLE
    assert information_regime(PolicyId.SUPERVISED_F1) is InformationRegime.SUPERVISED_DEVELOPMENT
    for policy in PolicyId:
        if policy is PolicyId.ORACLE_TEST:
            continue
        if policy in SUPERVISED_POLICIES:
            assert information_regime(policy) is InformationRegime.SUPERVISED_DEVELOPMENT
        else:
            assert information_regime(policy) is InformationRegime.BENIGN_ONLY


def test_deployable_policies_exclude_supervised_and_oracle() -> None:
    assert not is_deployable(PolicyId.ORACLE_TEST)
    for policy in SUPERVISED_POLICIES:
        assert not is_deployable(policy)
    assert is_deployable(PolicyId.REFERENCE_QUANTILE)
    assert is_deployable(PolicyId.FEDCRG)
