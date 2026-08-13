from fedcrg.domain.enums import ExperimentId
from fedcrg.experiments.dependencies import DependencyResolver


def test_dependency_order_includes_prerequisites() -> None:
    resolver = DependencyResolver()
    order = resolver.order((ExperimentId.DIAD_FEATURE_SENSITIVITY,))
    assert order.index(ExperimentId.PRIMARY_NBAIOT) < order.index(ExperimentId.EXTERNAL_DIAD)
    assert order.index(ExperimentId.EXTERNAL_DIAD) < order.index(
        ExperimentId.DIAD_FEATURE_SENSITIVITY
    )
