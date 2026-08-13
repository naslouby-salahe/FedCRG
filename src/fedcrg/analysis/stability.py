"""Split-sensitivity stability metrics for thresholds and deployment states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedcrg.core.enums import DecisionState


@dataclass(frozen=True, slots=True)
class ThresholdStability:
    count: int
    standard_deviation: float
    iqr: float
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class StateFrequency:
    state: DecisionState
    frequency: float


@dataclass(frozen=True, slots=True)
class StateStability:
    count: int
    transition_count: int
    transition_frequency: float
    state_frequencies: tuple[StateFrequency, ...]


def summarize_threshold_stability(values: tuple[float, ...]) -> ThresholdStability:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) == 0 or not np.isfinite(data).all():
        raise ValueError("Threshold stability requires finite non-empty values")
    return ThresholdStability(
        count=len(data),
        standard_deviation=float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        iqr=float(np.percentile(data, 75) - np.percentile(data, 25)),
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
    )


def summarize_state_stability(states: tuple[DecisionState, ...]) -> StateStability:
    if not states:
        raise ValueError("State stability requires at least one state")
    transitions = sum(
        left is not right
        for left, right in zip(states[:-1], states[1:], strict=True)
    )
    counts = {state: states.count(state) for state in DecisionState}
    total = len(states)
    return StateStability(
        count=total,
        transition_count=transitions,
        transition_frequency=transitions / max(1, total - 1),
        state_frequencies=tuple(
            StateFrequency(state, count / total) for state, count in counts.items()
        ),
    )
