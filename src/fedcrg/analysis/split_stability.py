"""Split-sensitivity stability of thresholds, deployment states, and repeated splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedcrg.analysis.descriptive_statistics import split_sensitivity_summary
from fedcrg.analysis.policy_contrasts import FederationResultRecord
from fedcrg.domain.enums import DecisionState, PolicyId
from fedcrg.domain.identifiers import ModelSeed


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
        left is not right for left, right in zip(states[:-1], states[1:], strict=True)
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


@dataclass(frozen=True, slots=True)
class SplitSensitivityRow:
    model_seed: ModelSeed
    policy: PolicyId
    metric: str
    median: float
    iqr: float
    p05: float
    p95: float
    calibration_split_count: int


def split_sensitivity(
    records: tuple[FederationResultRecord, ...],
) -> tuple[SplitSensitivityRow, ...]:
    """Summarize repeated role permutations without treating them as independent subjects."""

    grouped: dict[tuple[ModelSeed, PolicyId, str], list[float]] = {}
    for row in records:
        for metric in ("mebe", "high_excess", "attack_balanced_macro_tpr"):
            value = getattr(row, metric)
            if value is None:
                continue
            grouped.setdefault((row.model_seed, row.policy, metric), []).append(float(value))
    rows: list[SplitSensitivityRow] = []
    for (model_seed, policy, metric), values in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1].value, item[0][2]),
    ):
        summary = split_sensitivity_summary(tuple(values))
        rows.append(
            SplitSensitivityRow(
                model_seed=model_seed,
                policy=policy,
                metric=metric,
                median=summary.median,
                iqr=summary.iqr,
                p05=summary.p05,
                p95=summary.p95,
                calibration_split_count=len(values),
            )
        )
    return tuple(rows)
