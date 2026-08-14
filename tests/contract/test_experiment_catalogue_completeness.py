from fedcrg.config import Study
from fedcrg.types import ExperimentId


def test_all_experiment_ids_are_registered() -> None:
    assert {item.id for item in Study.load().catalogue.all()} == set(ExperimentId)
