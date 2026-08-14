"""Regression tests for experiment dependency ordering."""

from __future__ import annotations

from fedcrg.experiments.runner import DependencyResolver
from fedcrg.types import ExperimentId, ExperimentStatus


def test_dependency_order_matches_catalogue() -> None:
    resolver = DependencyResolver()
    order = resolver.order((ExperimentId.DIAD_FEATURE_SENSITIVITY,))
    assert order.index(ExperimentId.PRIMARY_NBAIOT) < order.index(ExperimentId.EXTERNAL_DIAD)
    assert order.index(ExperimentId.EXTERNAL_DIAD) < order.index(
        ExperimentId.DIAD_FEATURE_SENSITIVITY
    )


def test_blockers_identify_failed_dependencies() -> None:
    resolver = DependencyResolver()
    blockers = resolver.blockers(
        ExperimentId.READINESS_SAMPLE_SIZE,
        {ExperimentId.PRIMARY_NBAIOT: ExperimentStatus.FAILED},
    )
    assert blockers == (ExperimentId.PRIMARY_NBAIOT,)


def test_blockers_empty_when_dependencies_healthy() -> None:
    resolver = DependencyResolver()
    blockers = resolver.blockers(
        ExperimentId.READINESS_SAMPLE_SIZE,
        {ExperimentId.PRIMARY_NBAIOT: ExperimentStatus.COMPLETE},
    )
    assert blockers == ()
