"""Real-score sensitivity helpers for R2-R9 and R12."""

from __future__ import annotations

import numpy as np


def source_order_blocks(scores: np.ndarray, block_count: int = 5) -> tuple[np.ndarray, ...]:
    values = np.asarray(scores, dtype=np.float64)
    if block_count <= 0 or len(values) < block_count:
        raise ValueError("Source-order block analysis needs at least one row per block")
    return tuple(np.asarray(block, dtype=np.float64) for block in np.array_split(values, block_count))


def contaminate_benign_scores(
    benign: np.ndarray,
    attack_dev: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    values = np.asarray(benign, dtype=np.float64).copy()
    attacks = np.asarray(attack_dev, dtype=np.float64)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Contamination fraction must be in [0,1]")
    count = int(round(fraction * len(values)))
    if count == 0:
        return values
    if count > len(attacks):
        raise ValueError("Attack-development cache is too small for requested contamination")
    rng = np.random.Generator(np.random.PCG64(seed))
    target = rng.choice(len(values), size=count, replace=False)
    source = rng.choice(len(attacks), size=count, replace=False)
    values[target] = attacks[source]
    return values
