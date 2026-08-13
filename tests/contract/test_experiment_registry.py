from fedcrg.core.enums import ExperimentId
from fedcrg.experiments.registry import ExperimentRegistry


def test_all_experiment_ids_are_registered() -> None:
    assert {item.id for item in ExperimentRegistry().all()} == set(ExperimentId)
