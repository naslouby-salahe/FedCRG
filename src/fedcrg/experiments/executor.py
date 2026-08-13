"""Experiment execution with strict dependency and status semantics."""

from __future__ import annotations

from fedcrg.domain.enums import ExperimentId, ExperimentStatus
from fedcrg.experiments.dependencies import DependencyResolver
from fedcrg.experiments.lifecycle import assert_transition
from fedcrg.experiments.models import ExperimentExecution, ExperimentPlan, ExperimentRunner
from fedcrg.experiments.registry import ExperimentRegistry


class ExperimentExecutor:
    def __init__(self, registry: ExperimentRegistry | None = None) -> None:
        self.registry = registry or ExperimentRegistry()
        self.dependencies = DependencyResolver(self.registry)
        self.executions: dict[ExperimentId, ExperimentExecution] = {}

    def execute(self, plan: ExperimentPlan, runner: ExperimentRunner) -> ExperimentExecution:
        execution = ExperimentExecution(plan=plan)
        self.executions[plan.definition.id] = execution
        statuses = {experiment_id: item.status for experiment_id, item in self.executions.items()}
        if self.dependencies.blockers(plan.definition.id, statuses):
            execution.transition(ExperimentStatus.BLOCKED)
            return execution
        self._move(execution, ExperimentStatus.VALIDATING)
        self._move(execution, ExperimentStatus.READY)
        self._move(execution, ExperimentStatus.RUNNING)
        try:
            execution.result = runner(plan)
            self._move(execution, ExperimentStatus.VERIFYING)
            self._move(execution, ExperimentStatus.COMPLETE)
        except Exception as exc:  # boundary intentionally records arbitrary runner failures
            execution.error = f"{type(exc).__name__}: {exc}"
            if execution.status in {ExperimentStatus.RUNNING, ExperimentStatus.VERIFYING}:
                self._move(execution, ExperimentStatus.FAILED)
            else:
                execution.transition(ExperimentStatus.FAILED)
        return execution

    @staticmethod
    def _move(execution: ExperimentExecution, target: ExperimentStatus) -> None:
        assert_transition(execution.status, target)
        execution.transition(target)
