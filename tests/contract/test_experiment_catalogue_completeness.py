from fedcrg.domain.enums import ExperimentId
from fedcrg.experiments.experiment_definition import all_experiment_definitions


def test_all_experiment_ids_are_registered() -> None:
    assert {item.id for item in all_experiment_definitions()} == set(ExperimentId)
