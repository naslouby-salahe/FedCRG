"""Experiment plans, execution state, and allowed lifecycle transitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, TypeVar

from fedcrg.domain.enums import ExperimentStatus
from fedcrg.domain.identifiers import CalibrationSeed, ModelSeed, Sha256
from fedcrg.experiments.experiment_definition import ExperimentDefinition

TResult = TypeVar("TResult")

_ALLOWED_TRANSITIONS = {
    ExperimentStatus.PENDING: {ExperimentStatus.VALIDATING, ExperimentStatus.BLOCKED},
    ExperimentStatus.VALIDATING: {
        ExperimentStatus.READY,
        ExperimentStatus.INVALID,
        ExperimentStatus.FAILED,
    },
    ExperimentStatus.READY: {ExperimentStatus.RUNNING, ExperimentStatus.BLOCKED},
    ExperimentStatus.RUNNING: {ExperimentStatus.VERIFYING, ExperimentStatus.FAILED},
    ExperimentStatus.VERIFYING: {ExperimentStatus.COMPLETE, ExperimentStatus.FAILED},
    ExperimentStatus.COMPLETE: set(),
    ExperimentStatus.FAILED: set(),
    ExperimentStatus.BLOCKED: set(),
    ExperimentStatus.INVALID: set(),
}


def assert_transition(current: ExperimentStatus, target: ExperimentStatus) -> None:
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"Invalid experiment transition: {current.value} -> {target.value}")


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
