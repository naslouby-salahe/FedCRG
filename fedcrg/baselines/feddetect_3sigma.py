"""
FedDetect 3-Sigma Baseline

Implements B6 (FEDDETECT-3SIGMA) from Section 9.1.

Normative reference: Section 9.1, row B6
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class FedDetect3SigmaConfig:
    """
    Configuration for FedDetect 3-sigma baseline.
    
    Normative reference: Section 9.1, row B6
    """
    multiplier: float = 3.0  # 3 sigma
    ddof: int = 0  # Population std (ddof=0)


class FedDetect3SigmaBaseline:
    """
    Baseline B6: FEDDETECT-3SIGMA
    
    Published-style federated AE threshold baseline.
    
    Pool R+G+C scores.
    threshold = global mean + multiplier * sqrt(mean((s - mean)^2))
    where std uses ddof=0 (population standard deviation).
    
    Normative reference: Section 9.1, row B6
    """
    
    def __init__(self, config: FedDetect3SigmaConfig = None):
        """
        Initialize FedDetect 3-sigma baseline.
        
        Args:
            config: Baseline configuration
        """
        if config is None:
            config = FedDetect3SigmaConfig()
        self.config = config
    
    def compute_threshold(self, scores: np.ndarray) -> float:
        """
        Compute threshold from scores.
        
        Args:
            scores: Array of pooled R+G+C scores
            
        Returns:
            Threshold value
        """
        if len(scores) == 0:
            raise ValueError("Cannot compute threshold from empty scores")
        
        # Compute mean
        mean = float(np.mean(scores))
        
        # Compute population standard deviation (ddof=0)
        # std = sqrt(mean((s - mean)^2))
        variance = float(np.mean((scores - mean) ** 2))
        std = np.sqrt(variance)
        
        # Compute threshold
        threshold = mean + self.config.multiplier * std
        
        return threshold
    
    def compute_thresholds(
        self,
        client_pool_scores: Dict[str, np.ndarray],
    ) -> float:
        """
        Compute single threshold from pooled scores.
        
        For FedDetect 3-sigma, all clients use the same global threshold
        computed from pooled scores.
        
        Args:
            client_pool_scores: Dictionary mapping client_id to R+G+C scores
            
        Returns:
            Single threshold value for all clients
        """
        # Pool all scores
        all_scores = np.concatenate([
            scores for scores in client_pool_scores.values()
        ])
        
        return self.compute_threshold(all_scores)


# Singleton instance
B6_FEDDETECT_3SIGMA = FedDetect3SigmaBaseline()


def verify_feddetect_3sigma() -> None:
    """Verify FedDetect 3-sigma baseline."""
    baseline = FedDetect3SigmaBaseline()
    
    # Test with known values
    # If mean=0, std=1, threshold should be 3.0
    scores = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    threshold = baseline.compute_threshold(scores)
    
    mean = np.mean(scores)
    variance = np.mean((scores - mean) ** 2)
    std = np.sqrt(variance)
    expected = mean + 3.0 * std
    
    assert abs(threshold - expected) < 1e-10, f"Expected {expected}, got {threshold}"
    
    # Test with all zeros
    scores_zeros = np.array([0.0, 0.0, 0.0])
    threshold_zeros = baseline.compute_threshold(scores_zeros)
    assert threshold_zeros == 0.0, f"Expected 0.0 for all zeros, got {threshold_zeros}"
    
    # Test pooling
    client_scores = {
        "nb01": np.array([1.0, 2.0, 3.0]),
        "nb02": np.array([4.0, 5.0, 6.0]),
    }
    pooled_threshold = baseline.compute_thresholds(client_scores)
    
    all_scores = np.concatenate([client_scores[cid] for cid in client_scores])
    expected_pooled = baseline.compute_threshold(all_scores)
    
    assert abs(pooled_threshold - expected_pooled) < 1e-10
    
    print("FedDetect 3-sigma baseline verification passed.")


if __name__ == "__main__":
    verify_feddetect_3sigma()
