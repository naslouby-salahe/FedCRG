"""Descriptive summaries of pre-registered fixed-federation evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DescriptiveSummary:
    values: tuple[float, ...]
    mean: float
    standard_deviation: float
    median: float
    minimum: float
    maximum: float


def describe(values: tuple[float, ...]) -> DescriptiveSummary:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) == 0 or not np.isfinite(data).all():
        raise ValueError("Summary values must be finite and non-empty")
    return DescriptiveSummary(
        values=values,
        mean=float(np.mean(data)),
        standard_deviation=float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        median=float(np.median(data)),
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
    )


def split_sensitivity_summary(values: tuple[float, ...]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0:
        raise ValueError("Split sensitivity requires values")
    return {
        "median": float(np.median(data)),
        "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
        "p05": float(np.percentile(data, 5)),
        "p95": float(np.percentile(data, 95)),
    }
