from fedcrg.core.enums import ExperimentId, ExperimentStatus
from fedcrg.core.ids import CalibrationSeed, ModelSeed, Sha256
from fedcrg.experiments.dependencies import DependencyResolver
from fedcrg.experiments.executor import ExperimentExecutor
from fedcrg.experiments.models import ExperimentPlan
from fedcrg.experiments.registry import ExperimentRegistry


def _plan(experiment_id: ExperimentId) -> ExperimentPlan:
    registry = ExperimentRegistry()
    return ExperimentPlan(
        registry.get(experiment_id), Sha256("a" * 64), ModelSeed(11), CalibrationSeed(1000)
    )


def test_primary_execution_completes() -> None:
    executor = ExperimentExecutor()
    execution = executor.execute(_plan(ExperimentId.PRIMARY_NBAIOT), lambda plan: {"ok": True})
    assert execution.status is ExperimentStatus.COMPLETE
    assert execution.result == {"ok": True}


def test_runner_failure_is_recorded() -> None:
    executor = ExperimentExecutor()

    def fail(plan: ExperimentPlan) -> None:
        raise RuntimeError("boom")

    execution = executor.execute(_plan(ExperimentId.PRIMARY_NBAIOT), fail)
    assert execution.status is ExperimentStatus.FAILED
    assert execution.error is not None
    assert "boom" in execution.error


def test_dependant_is_blocked_until_dependency_completed() -> None:
    executor = ExperimentExecutor()
    execution = executor.execute(_plan(ExperimentId.READINESS_SAMPLE_SIZE), lambda plan: None)
    assert execution.status is ExperimentStatus.BLOCKED


def test_dependency_order_includes_prerequisites() -> None:
    resolver = DependencyResolver(ExperimentRegistry())
    order = resolver.order((ExperimentId.DIAD_FEATURE_SENSITIVITY,))
    assert order.index(ExperimentId.PRIMARY_NBAIOT) < order.index(ExperimentId.EXTERNAL_DIAD)
    assert order.index(ExperimentId.EXTERNAL_DIAD) < order.index(
        ExperimentId.DIAD_FEATURE_SENSITIVITY
    )
