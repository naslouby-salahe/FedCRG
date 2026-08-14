from fedcrg.types import DataRole, DecisionState, ExperimentStatus, PolicyId


def test_closed_domains_have_no_duplicate_values() -> None:
    for enum_type in (DataRole, DecisionState, ExperimentStatus, PolicyId):
        values = [member.value for member in enum_type]
        assert len(values) == len(set(values))
