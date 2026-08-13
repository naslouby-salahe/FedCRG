"""Typed experiment catalogue, parameter axes, plans, and execution state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Generic, TypeVar

from fedcrg.core.enums import ArtifactType, ExperimentId, ExperimentStatus, ExperimentType, PolicyId
from fedcrg.core.ids import Sha256

TResult = TypeVar("TResult")
Scalar = int | float | str


@dataclass(frozen=True, slots=True)
class ParameterAxis:
    name: str
    values: tuple[Scalar, ...]


@dataclass(frozen=True, slots=True)
class WorkloadExpectation:
    monte_carlo_trials: int = 0
    exact_cells: int = 0
    detector_trainings: int = 0


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    id: ExperimentId
    protocol_code: str
    type: ExperimentType
    axes: tuple[ParameterAxis, ...] = ()
    dependencies: tuple[ExperimentId, ...] = ()
    policies: tuple[PolicyId, ...] = ()
    required_artifacts: tuple[ArtifactType, ...] = ()
    workload: WorkloadExpectation = WorkloadExpectation()
    confirmatory: bool = False
    description: str = ""


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    definition: ExperimentDefinition
    config_hash: Sha256
    model_seed: int
    calibration_seed: int


@dataclass(slots=True)
class ExperimentExecution(Generic[TResult]):
    plan: ExperimentPlan
    status: ExperimentStatus = ExperimentStatus.PENDING
    result: TResult | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def transition(self, status: ExperimentStatus) -> None:
        self.status = status
        now = datetime.now(timezone.utc).isoformat()
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
