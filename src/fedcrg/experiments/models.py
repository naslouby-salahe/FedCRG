"""Typed experiment catalogue, parameter grids, plans, and execution state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Generic, TypeAlias, TypeVar
from collections.abc import Callable

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
    """An independent axis whose values may be crossed with other independent axes."""

    id: ExperimentAxisId
    values: tuple[AxisValue, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"Experiment axis {self.id.value} must contain values")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"Experiment axis {self.id.value} contains duplicate values")


@dataclass(frozen=True, slots=True)
class ParameterSetting:
    axis: ExperimentAxisId
    value: AxisValue


@dataclass(frozen=True, slots=True)
class ParameterCell:
    """A coupled combination that must be evaluated together, not Cartesian-crossed."""

    settings: tuple[ParameterSetting, ...]

    def __post_init__(self) -> None:
        if not self.settings:
            raise ValueError("A parameter cell must contain at least one setting")
        axes = tuple(item.axis for item in self.settings)
        if len(set(axes)) != len(axes):
            raise ValueError("A parameter cell cannot assign the same axis twice")

    def value(self, axis: ExperimentAxisId) -> AxisValue:
        for setting in self.settings:
            if setting.axis is axis:
                return setting.value
        raise KeyError(axis.value)


@dataclass(frozen=True, slots=True)
class WorkloadExpectation:
    monte_carlo_trials: int = 0
    exact_cells: int = 0
    detector_trainings: int = 0

    def __post_init__(self) -> None:
        if min(self.monte_carlo_trials, self.exact_cells, self.detector_trainings) < 0:
            raise ValueError("Workload expectations cannot be negative")


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    id: ExperimentId
    protocol_code: ExperimentCode
    type: ExperimentType
    axes: tuple[ParameterAxis, ...] = ()
    coupled_cells: tuple[ParameterCell, ...] = ()
    dependencies: tuple[ExperimentId, ...] = ()
    policies: tuple[PolicyId, ...] = ()
    required_artifacts: tuple[ArtifactType, ...] = ()
    workload: WorkloadExpectation = WorkloadExpectation()
    confirmatory: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        independent_axes = tuple(axis.id for axis in self.axes)
        if len(set(independent_axes)) != len(independent_axes):
            raise ValueError(f"Duplicate axis in {self.protocol_code.value}")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"Duplicate dependency in {self.protocol_code.value}")
        if self.id in self.dependencies:
            raise ValueError(f"Experiment {self.protocol_code.value} cannot depend on itself")
        if len(set(self.policies)) != len(self.policies):
            raise ValueError(f"Duplicate policy in {self.protocol_code.value}")

    def axis(self, axis_id: ExperimentAxisId) -> ParameterAxis:
        for item in self.axes:
            if item.id is axis_id:
                return item
        raise KeyError(
            f"Experiment {self.protocol_code.value} has no independent {axis_id.value} axis"
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
        now = datetime.now(UTC)
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
