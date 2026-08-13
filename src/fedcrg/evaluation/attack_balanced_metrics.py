"""Attack-group-balanced utility metrics."""

from __future__ import annotations

import numpy as np


def attack_balanced_tpr(
    scores: np.ndarray,
    labels: np.ndarray,
    attack_groups: np.ndarray,
    threshold: float,
) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    groups_array = np.asarray(attack_groups, dtype=object)
    groups = sorted(set(groups_array[targets == 1].astype(str)))
    if not groups:
        return None
    per_group: list[float] = []
    for group in groups:
        mask = (targets == 1) & (groups_array.astype(str) == group)
        if not np.any(mask):
            continue
        per_group.append(float(np.mean(values[mask] > threshold)))
    return float(np.mean(per_group)) if per_group else None
