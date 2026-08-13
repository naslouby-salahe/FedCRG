"""Typed experiment catalogue, parameter axes, plans, and execution state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Generic, TypeAlias, TypeVar

from fedcrg.core.enums import (
    ArtifactType,
    CalibrationAssignmentMode,
    ContaminationDirection,
    ExperimentAxisId,
    ExperimentCode,
    ExperimentId,
    ExperimentStatus,
    ExperimentType,
    MultiplicityProcedure,
    PolicyId,
    SyntheticDistribution,
)
from fedcrg.core.ids import CalibrationSeed, ModelSeed, Sha256

TResult = TypeVar("TResult")
AxisValue: TypeAlias = (
    int
    | float
    | SyntheticDistribution
    | ContaminationDirection
    | MultiplicityProcedure
    | CalibrationAssignmentMode
)


@dataclass(frozen=True, slots=True)
class ParameterAxis:
    id: ExperimentAxisId
    values: tuple[AxisValue, ...]


@dataclass(frozen=True, slots=True)
class WorkloadExpectation:
    monte_carlo_trials: int = 0
    exact_cells: int = 0
    detector_trainings: int = 0


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    id: ExperimentId
    protocol_code: ExperimentCode
    type: ExperimentType
    axes: tuple[ParameterAxis, ...] = ()
    dependencies: tuple[ExperimentId, ...] = ()
    policies: tuple[PolicyId, ...] = ()
    required_artifacts: tuple[ArtifactType, ...] = ()
    workload: WorkloadExpectation = WorkloadExpectation()
    confirmatory: bool = False
    description: str = ""

    def axis(self, axis_id: ExperimentAxisId) -> ParameterAxis:
        for item in self.axes:
            if item.id is axis_id:
                return item
        raise KeyError(
            f"Experiment {self.protocol_code.value} has no {axis_id.value} axis"
        )


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    definition: ExperimentDefinition
    config_hash: Sha256
    model_seed: ModelSeed
    calibration_seed: CalibrationSeed


@dataclass(slots=True)
class ExperimentExecution(Generic[TResult]):
    plan: ExperimentPlan
    status: ExperimentStatus = ExperimentStatus.PENDING
    result: TResult | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def transition(self, status: ExperimentStatus) -> None:
        from fedcrg.experiments.lifecycle import assert_transition

        assert_transition(self.status, status)
        self.status = status
        now = datetime.now(timezone.utc)
        if status is ExperimentStatus.RUNNING:
            self.started_at = now
        if status in {
            ExperimentStatus.COMPLETE,
            ExperimentStatus.FAILED,
            ExperimentStatus.BLOCKED,
            ExperimentStatus.INVALID,
        }:
            self.finished_at = now


ExperimentRunner = Callable[[ExperimentPlan], TResult]
