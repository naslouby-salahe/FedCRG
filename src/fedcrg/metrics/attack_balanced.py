"""Attack-balanced detection metrics."""
import numpy as np
def attack_balanced_tpr(scores: np.ndarray, labels: np.ndarray, attack_groups: np.ndarray, threshold: float) -> float:
    groups = sorted(set(attack_groups[labels == 1].astype(str)))
    if not groups: return 0.0
    values=[]
    for group in groups:
        mask=(labels==1)&(attack_groups.astype(str)==group); values.append(float(np.mean(scores[mask] > threshold)))
    return float(np.mean(values))
