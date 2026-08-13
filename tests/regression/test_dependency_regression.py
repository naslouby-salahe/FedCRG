from fedcrg.domain.enums import ExperimentId, ExperimentStatus
from fedcrg.experiments.dependencies import DependencyResolver
from fedcrg.experiments.registry import ExperimentRegistry


def test_failed_dependency_blocks_dependant() -> None:
    resolver = DependencyResolver(ExperimentRegistry())
    statuses = {ExperimentId.PRIMARY_NBAIOT: ExperimentStatus.FAILED}
    blockers = resolver.blockers(ExperimentId.READINESS_SAMPLE_SIZE, statuses)
    assert blockers == (ExperimentId.PRIMARY_NBAIOT,)
