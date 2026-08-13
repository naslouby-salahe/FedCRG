"""Experiment plans and allowed lifecycle transitions."""

from __future__ import annotations

from dataclasses import dataclass

from fedcrg.domain.enums import ExperimentStatus
from fedcrg.domain.identifiers import CalibrationSeed, ModelSeed, Sha256
from fedcrg.experiments.experiment_definition import ExperimentDefinition

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
