"""Experiment definitions and execution results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Generic, TypeVar

from fedcrg.core.enums import ArtifactType, ExperimentId, ExperimentStatus, ExperimentType

TResult = TypeVar("TResult")


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    id: ExperimentId
    type: ExperimentType
    dependencies: tuple[ExperimentId, ...] = ()
    required_artifacts: tuple[ArtifactType, ...] = ()
    confirmatory: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    definition: ExperimentDefinition
    config_hash: str
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
