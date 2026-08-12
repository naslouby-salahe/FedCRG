"""Allowed experiment lifecycle transitions."""

from fedcrg.core.enums import ExperimentStatus

_ALLOWED = {
    ExperimentStatus.PENDING: {ExperimentStatus.VALIDATING, ExperimentStatus.BLOCKED},
    ExperimentStatus.VALIDATING: {ExperimentStatus.READY, ExperimentStatus.INVALID, ExperimentStatus.FAILED},
    ExperimentStatus.READY: {ExperimentStatus.RUNNING, ExperimentStatus.BLOCKED},
    ExperimentStatus.RUNNING: {ExperimentStatus.VERIFYING, ExperimentStatus.FAILED},
    ExperimentStatus.VERIFYING: {ExperimentStatus.COMPLETE, ExperimentStatus.FAILED},
    ExperimentStatus.COMPLETE: set(),
    ExperimentStatus.FAILED: set(),
    ExperimentStatus.BLOCKED: set(),
    ExperimentStatus.INVALID: set(),
}


def assert_transition(current: ExperimentStatus, target: ExperimentStatus) -> None:
    if target not in _ALLOWED[current]:
        raise ValueError(f"Invalid experiment transition: {current.value} -> {target.value}")
