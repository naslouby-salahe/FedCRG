"""Pre-registered fixed-federation statistical summaries."""

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


@dataclass(frozen=True, slots=True)
class PairedBootstrapInterval:
    observed_difference: float
    lower: float
    upper: float
    replicates: int
    seed: int


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


def paired_model_seed_bootstrap(
    method: tuple[float, ...],
    comparator: tuple[float, ...],
    *,
    replicates: int = 10000,
    seed: int = 424242,
) -> PairedBootstrapInterval:
    """Bootstrap paired model-seed indices, never treat calibration splits as subjects."""

    left = np.asarray(method, dtype=np.float64)
    right = np.asarray(comparator, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) == 0:
        raise ValueError("Paired bootstrap inputs must be aligned non-empty vectors")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(left), size=(replicates, len(left)))
    differences = np.mean(left[indices] - right[indices], axis=1)
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return PairedBootstrapInterval(
        observed_difference=float(np.mean(left - right)),
        lower=float(lower),
        upper=float(upper),
        replicates=replicates,
        seed=seed,
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
