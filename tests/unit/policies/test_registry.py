from fedcrg.domain.enums import PolicyId
from fedcrg.policies.registry import InformationRegime, PolicyRegistry


def test_policy_registry_is_complete_and_unique() -> None:
    registry = PolicyRegistry()
    assert set(registry.all_ids()) == set(PolicyId)
    assert len(registry.all_ids()) == len(PolicyId)
    assert (
        registry.get(PolicyId.ORACLE_TEST).information_regime is InformationRegime.FINAL_TEST_ORACLE
    )
    assert (
        registry.get(PolicyId.SUPERVISED_F1).information_regime
        is InformationRegime.SUPERVISED_DEVELOPMENT
    )
