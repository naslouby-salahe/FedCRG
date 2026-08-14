from __future__ import annotations

from fedcrg.experiments.runner import DependencyResolver
from fedcrg.types import ExperimentId, ExperimentStatus


def test_order_matches_catalogue_dependencies() -> None:
    resolver = DependencyResolver()
    order = resolver.order((ExperimentId.DIAD_FEATURE_SENSITIVITY,))
    assert order.index(ExperimentId.PRIMARY_NBAIOT) < order.index(ExperimentId.EXTERNAL_DIAD)
    assert order.index(ExperimentId.EXTERNAL_DIAD) < order.index(
        ExperimentId.DIAD_FEATURE_SENSITIVITY
    )


def test_blockers_detect_failed_dependencies() -> None:
    resolver = DependencyResolver()
    blockers = resolver.blockers(
        ExperimentId.READINESS_SAMPLE_SIZE,
        {ExperimentId.PRIMARY_NBAIOT: ExperimentStatus.FAILED},
    )
    assert blockers == (ExperimentId.PRIMARY_NBAIOT,)


def test_no_blockers_when_dependencies_complete() -> None:
    resolver = DependencyResolver()
    blockers = resolver.blockers(
        ExperimentId.READINESS_SAMPLE_SIZE,
        {ExperimentId.PRIMARY_NBAIOT: ExperimentStatus.COMPLETE},
    )
    assert blockers == ()
