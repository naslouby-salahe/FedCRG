"""Paired model-seed bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PairedBootstrapInterval:
    observed_difference: float
    lower: float
    upper: float
    replicates: int
    seed: int


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
