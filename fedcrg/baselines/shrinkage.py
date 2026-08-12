"""
Shrinkage Baseline

Implements B5 (SHRINKAGE) from Section 9.2.

Normative reference: Section 9.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fedcrg.reference import Alpha
from fedcrg.baselines.quantile import QuantileBaseline, QuantileBaselineConfig, compute_quantile_threshold


@dataclass(frozen=True, slots=True)
class ShrinkageConfig:
    """
    Configuration for shrinkage baseline.
    
    Normative reference: Section 9.2
    """
    alpha: float = Alpha()
    n0_candidates: List[int] = field(default_factory=lambda: [100, 300, 1000, 3000, 10000])
    
    # Precomputed expected ranks from Section 9.1
    # These are used for verification
    nbaiot_local_q99_full_n: int = 5500
    nbaiot_local_q99_full_q: int = 5446


class ShrinkageBaseline:
    """
    Baseline B5: SHRINKAGE
    
    Required due to adjacent shrinkage literature.
    
    tau_shr = w * tau_local,Q99 + (1-w) * tau_ref
    where w = n_C / (n_C + n0)
    
    Candidate n0 grid: {100, 300, 1000, 3000, 10000}
    
    For each n0:
    - Estimate each client FPR on G_k
    - Compute mean absolute target-FPR error across clients
    - Choose n0 with minimum mean error
    - Ties choose the largest n0 (more pooling)
    
    No attack data and no final test data are used.
    
    Normative reference: Section 9.2
    """
    
    def __init__(
        self,
        config: ShrinkageConfig,
        tau_ref: float,
    ):
        """
        Initialize shrinkage baseline.
        
        Args:
            config: Shrinkage configuration
            tau_ref: Federation reference threshold
        """
        self.config = config
        self.tau_ref = tau_ref
        
        # Quantile baseline for computing tau_local,Q99
        self.quantile = QuantileBaseline(QuantileBaselineConfig(alpha=config.alpha))
    
    def compute_weight(self, n_c: int, n0: int) -> float:
        """
        Compute shrinkage weight.
        
        w(n0) = n_C / (n_C + n0)
        
        Args:
            n_c: Size of local calibration set
            n0: Shrinkage parameter
            
        Returns:
            Weight for local threshold
        """
        return n_c / (n_c + n0) if (n_c + n0) > 0 else 0.0
    
    def compute_shrinkage_threshold(
        self,
        calibration_scores: np.ndarray,
        gate_scores: np.ndarray,
        n0: int,
    ) -> float:
        """
        Compute shrinkage threshold for a client.
        
        Args:
            calibration_scores: Array of C_k scores (for tau_local,Q99)
            gate_scores: Array of G_k scores (for FPR estimation)
            n0: Shrinkage parameter
            
        Returns:
            Shrinkage threshold
        """
        n_c = len(calibration_scores)
        
        # Compute tau_local,Q99
        tau_local_q99 = self.quantile.compute_threshold(calibration_scores)
        
        # Compute weight
        w = self.compute_weight(n_c, n0)
        
        # Compute shrinkage threshold
        tau_shr = w * tau_local_q99 + (1 - w) * self.tau_ref
        
        return tau_shr
    
    def estimate_fpr_error(
        self,
        calibration_scores: np.ndarray,
        gate_scores: np.ndarray,
        threshold: float,
        target_alpha: float = Alpha(),
    ) -> float:
        """
        Estimate FPR error on G_k.
        
        Args:
            calibration_scores: Array of C_k scores (for threshold fitting)
            gate_scores: Array of G_k scores (for FPR estimation)
            threshold: Threshold to evaluate
            target_alpha: Target FPR (default 0.01)
            
        Returns:
            Absolute error: |estimated FPR - target_alpha|
        """
        # Estimate FPR as fraction of gate scores exceeding threshold
        if len(gate_scores) > 0:
            fpr_estimate = float(np.mean(gate_scores > threshold))
        else:
            fpr_estimate = 0.0
        
        return abs(fpr_estimate - target_alpha)
    
    def select_best_n0(
        self,
        client_calibration_scores: Dict[str, np.ndarray],
        client_gate_scores: Dict[str, np.ndarray],
    ) -> Tuple[int, float, Dict[int, float]]:
        """
        Select best n0 from candidate grid.
        
        For each n0:
        - Compute shrinkage threshold for each client
        - Estimate FPR on G_k for each client
        - Compute mean absolute error across clients
        
        Returns n0 with minimum mean error. Ties choose largest n0.
        
        Args:
            client_calibration_scores: Dictionary mapping client_id to C_k scores
            client_gate_scores: Dictionary mapping client_id to G_k scores
            
        Returns:
            Tuple of (best_n0, min_mean_error, n0_errors)
            where n0_errors maps n0 to mean error
        """
        n0_errors = {}
        
        for n0 in self.config.n0_candidates:
            total_error = 0.0
            n_clients = 0
            
            for client_id, calibration_scores in client_calibration_scores.items():
                if client_id in client_gate_scores:
                    gate_scores = client_gate_scores[client_id]
                    
                    threshold = self.compute_shrinkage_threshold(
                        calibration_scores, gate_scores, n0
                    )
                    
                    error = self.estimate_fpr_error(
                        calibration_scores, gate_scores, threshold
                    )
                    
                    total_error += error
                    n_clients += 1
            
            mean_error = total_error / n_clients if n_clients > 0 else float('inf')
            n0_errors[n0] = mean_error
        
        # Find best n0 (minimum mean error, ties go to largest n0)
        best_n0 = None
        best_error = float('inf')
        
        for n0 in sorted(self.config.n0_candidates, reverse=True):
            error = n0_errors[n0]
            if error <= best_error:
                best_n0 = n0
                best_error = error
        
        return best_n0, best_error, n0_errors
    
    def compute_threshold(
        self,
        calibration_scores: np.ndarray,
        gate_scores: np.ndarray,
        best_n0: Optional[int] = None,
        client_id: Optional[str] = None,
    ) -> float:
        """
        Compute threshold for a client.
        
        If best_n0 is not provided, uses the first candidate (for testing).
        In production, select_best_n0 should be called first.
        
        Args:
            calibration_scores: Array of C_k scores
            gate_scores: Array of G_k scores
            best_n0: Best shrinkage parameter (optional)
            client_id: Optional client identifier
            
        Returns:
            Threshold value
        """
        if best_n0 is None:
            best_n0 = self.config.n0_candidates[0]
        
        return self.compute_shrinkage_threshold(
            calibration_scores, gate_scores, best_n0
        )


# Singleton instance
B5_SHRINKAGE = None


def select_best_n0(
    client_calibration_scores: Dict[str, np.ndarray],
    client_gate_scores: Dict[str, np.ndarray],
    tau_ref: float,
    config: Optional[ShrinkageConfig] = None,
) -> Tuple[int, float]:
    """
    Select best n0 for shrinkage baseline.
    
    Convenience function that creates a ShrinkageBaseline and selects n0.
    
    Args:
        client_calibration_scores: Dictionary mapping client_id to C_k scores
        client_gate_scores: Dictionary mapping client_id to G_k scores
        tau_ref: Federation reference threshold
        config: Optional configuration
        
    Returns:
        Tuple of (best_n0, best_mean_error)
    """
    if config is None:
        config = ShrinkageConfig()
    
    baseline = ShrinkageBaseline(config, tau_ref)
    best_n0, best_error, _ = baseline.select_best_n0(
        client_calibration_scores, client_gate_scores
    )
    
    return best_n0, best_error


def verify_shrinkage() -> None:
    """Verify shrinkage baseline."""
    # Create baseline
    baseline = ShrinkageBaseline(
        ShrinkageConfig(),
        tau_ref=1.0,
    )
    
    # Test weight computation
    w = baseline.compute_weight(n_c=2000, n0=1000)
    expected_w = 2000 / 3000
    assert abs(w - expected_w) < 1e-10, f"Weight computation: expected {expected_w}, got {w}"
    
    # Test shrinkage threshold
    np.random.seed(42)
    calibration_scores = np.random.randn(2000)
    gate_scores = np.random.randn(3000)
    
    threshold = baseline.compute_shrinkage_threshold(
        calibration_scores, gate_scores, n0=1000
    )
    
    # Threshold should be between tau_local and tau_ref
    tau_local = baseline.quantile.compute_threshold(calibration_scores)
    assert tau_local < threshold < baseline.tau_ref or threshold < tau_local < baseline.tau_ref
    
    # Test FPR error estimation
    error = baseline.estimate_fpr_error(
        calibration_scores, gate_scores, threshold
    )
    assert error >= 0.0
    
    # Test n0 selection
    client_calibration = {"nb01": calibration_scores}
    client_gate = {"nb01": gate_scores}
    
    best_n0, best_error, n0_errors = baseline.select_best_n0(
        client_calibration, client_gate
    )
    
    assert best_n0 in baseline.config.n0_candidates
    assert best_error >= 0.0
    
    print("Shrinkage baseline verification passed.")


if __name__ == "__main__":
    verify_shrinkage()
