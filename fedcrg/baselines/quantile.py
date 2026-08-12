"""
Quantile Baselines

Implements quantile-based baselines B0, B1, B2, B4 from Section 9.1.

Normative reference: Section 9.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fedcrg.data.base import DatasetRole
from fedcrg.reference import Alpha, A, B


@dataclass(frozen=True, slots=True)
class QuantileBaselineConfig:
    """
    Configuration for quantile baselines.
    
    Normative reference: Section 9.1
    """
    alpha: float = Alpha()  # 0.01
    use_strict_inequality: bool = True  # anomaly iff score > threshold


# Precomputed quantile ranks from Section 9.1 table
# N-BaIoT
NBAIOT_REF_Q99_R_N = 4500  # 9 * 500
NBAIOT_REF_Q99_R_Q = 4456

NBAIOT_GLOBAL_Q99_FULL_N = 49500  # 9 * 5500
NBAIOT_GLOBAL_Q99_FULL_Q = 49006

NBAIOT_LOCAL_Q99_FULL_N = 5500  # per client
NBAIOT_LOCAL_Q99_FULL_Q = 5446

NBAIOT_B4_Q99_N = 2000  # per client, from C_k
NBAIOT_B4_Q99_Q = 1981

# DIAD
# DIAD values depend on K_D (number of eligible clients)
# For K_D=105:
DIAD_REF_Q99_R_N = 31500  # 105 * 300
DIAD_REF_Q99_R_Q = 31186  # q(31500, 0.01)

DIAD_GLOBAL_Q99_FULL_N = 346500  # 105 * 3300
DIAD_GLOBAL_Q99_FULL_Q = 343036

DIAD_LOCAL_Q99_FULL_N = 3300  # per client
DIAD_LOCAL_Q99_FULL_Q = 3268

DIAD_B4_Q99_N = 1500  # per client, from C_k
DIAD_B4_Q99_Q = 1486


def compute_quantile_rank(n: int, alpha: float = Alpha()) -> int:
    """
    Compute quantile rank using the formula from Section 9.1:
    q(N, alpha) = min(N, ceil((N + 1)(1 - alpha)))
    
    Args:
        n: Number of samples
        alpha: Target FPR (default 0.01)
        
    Returns:
        Quantile rank (1-indexed)
        
    Normative reference: Section 9.1
    """
    # Compute (N + 1) * (1 - alpha)
    value = (n + 1) * (1 - alpha)
    
    # Ceiling
    rank = math.ceil(value)
    
    # Clamp to [0, N]
    rank = min(n, max(0, rank))
    
    return int(rank)


def compute_quantile_threshold(
    scores: np.ndarray,
    alpha: float = Alpha(),
) -> float:
    """
    Compute quantile threshold from scores.
    
    Sorts scores in ascending order and selects the q-th element
    where q = min(N, ceil((N + 1)(1 - alpha))).
    
    Args:
        scores: Array of anomaly scores
        alpha: Target FPR (default 0.01)
        
    Returns:
        Threshold value
        
    Normative reference: Section 9.1
    """
    if len(scores) == 0:
        raise ValueError("Cannot compute threshold from empty scores")
    
    # Sort in ascending order
    sorted_scores = np.sort(scores)
    
    # Compute rank
    n = len(sorted_scores)
    rank = compute_quantile_rank(n, alpha)
    
    # Get threshold (1-indexed, so index is rank-1)
    # But the formula uses 1-indexed ranks
    # In 0-indexed numpy array, the q-th element is at index q-1
    threshold = float(sorted_scores[rank - 1]) if rank > 0 else float(sorted_scores[0])
    
    return threshold


class QuantileBaseline:
    """
    Base class for quantile-based baselines.
    
    All quantile baselines use the same deterministic rank convention.
    
    Normative reference: Section 9.1
    """
    
    def __init__(self, config: QuantileBaselineConfig):
        """
        Initialize quantile baseline.
        
        Args:
            config: Baseline configuration
        """
        self.config = config
    
    def compute_threshold(self, scores: np.ndarray) -> float:
        """
        Compute threshold from scores.
        
        Args:
            scores: Array of anomaly scores
            
        Returns:
            Threshold value
        """
        return compute_quantile_threshold(scores, self.config.alpha)
    
    def compute_thresholds(
        self,
        client_scores: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute thresholds for multiple clients.
        
        Args:
            client_scores: Dictionary mapping client_id to scores
            
        Returns:
            Dictionary mapping client_id to threshold
        """
        return {
            client_id: self.compute_threshold(scores)
            for client_id, scores in client_scores.items()
        }


class B0_REF_Q99_R(QuantileBaseline):
    """
    Baseline B0: REF-Q99-R
    
    Benign only baseline. Threshold from R only.
    tau_ref from R only (not presented as strongest shared baseline).
    
    Uses pooled R_k with equal per-client counts.
    q = min(N, ceil((N+1)(1-alpha)))
    anomaly iff score > threshold (strict inequality)
    
    Normative reference: Section 9.1, row B0
    """
    
    # Precomputed values for verification
    # N-BaIoT: 9 clients * 500 = 4500, q = 4456
    # DIAD: K_D * 300, q = min(300*K_D, ceil((300*K_D+1)*0.99))
    
    pass  # Inherits everything from QuantileBaseline


class B1_GLOBAL_Q99_FULL(QuantileBaseline):
    """
    Baseline B1: GLOBAL-Q99-FULL
    
    Benign only baseline. Strong full benign-policy-budget always-shared.
    Pool every client R+G+C (5,500 N-BaIoT; 3,300 DIAD) with equal per-client counts.
    q = min(N, ceil((N+1)(1-alpha)))
    anomaly iff score > threshold (strict inequality)
    
    Normative reference: Section 9.1, row B1
    """
    pass


class B2_LOCAL_Q99_FULL(QuantileBaseline):
    """
    Baseline B2: LOCAL-Q99-FULL
    
    Benign only baseline. Strong full benign-policy-budget always-local.
    Per client R+G+C.
    q = min(n, ceil((n+1)(1-alpha)))
    anomaly iff score > threshold (strict inequality)
    
    Normative reference: Section 9.1, row B2
    """
    pass


class B4_GATE_B_ONLY(QuantileBaseline):
    """
    Baseline B4: GATE-B-ONLY
    
    Benign only baseline. Ablates finite-sample readiness.
    If Gate B mismatch, use C_(q_C) with q_C = min(n_C, ceil((n_C+1)(1-alpha)))
    Otherwise use tau_ref.
    
    This baseline requires Gate B computation to determine which threshold to use.
    
    Normative reference: Section 9.1, row B4
    """
    
    def __init__(
        self,
        config: QuantileBaselineConfig,
        gate_b_states: Dict[str, str],
        tau_ref: float,
    ):
        """
        Initialize B4 baseline.
        
        Args:
            config: Baseline configuration
            gate_b_states: Dictionary mapping client_id to Gate B state
                         ("LOW", "HIGH", or "NONE")
            tau_ref: Federation reference threshold
        """
        super().__init__(config)
        self.gate_b_states = gate_b_states
        self.tau_ref = tau_ref
    
    def compute_threshold(
        self,
        scores: np.ndarray,
        client_id: Optional[str] = None,
    ) -> float:
        """
        Compute threshold for a client.
        
        If Gate B mismatch (LOW or HIGH), use quantile from C_k.
        Otherwise use tau_ref.
        
        Args:
            scores: Array of C_k scores
            client_id: Client identifier
            
        Returns:
            Threshold value
        """
        if client_id and client_id in self.gate_b_states:
            state = self.gate_b_states[client_id]
            if state in ("LOW", "HIGH"):
                return super().compute_threshold(scores)
        
        return self.tau_ref


def verify_quantile() -> None:
    """Verify quantile computations."""
    # Test formula
    # For N=4500, alpha=0.01:
    # (4500 + 1) * (1 - 0.01) = 4501 * 0.99 = 4455.99
    # ceil(4455.99) = 4456
    # min(4500, 4456) = 4456
    
    n = 4500
    q = compute_quantile_rank(n, alpha=0.01)
    assert q == 4456, f"Expected q=4456 for n=4500, got {q}"
    
    # For N=5500:
    # (5500 + 1) * 0.99 = 5501 * 0.99 = 5445.99
    # ceil = 5446
    n = 5500
    q = compute_quantile_rank(n, alpha=0.01)
    assert q == 5446, f"Expected q=5446 for n=5500, got {q}"
    
    # For N=2000:
    # (2000 + 1) * 0.99 = 2001 * 0.99 = 1980.99
    # ceil = 1981
    n = 2000
    q = compute_quantile_rank(n, alpha=0.01)
    assert q == 1981, f"Expected q=1981 for n=2000, got {q}"
    
    # Test threshold computation
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    threshold = compute_quantile_threshold(scores, alpha=0.01)
    # For 5 scores, q = min(5, ceil(6*0.99)) = min(5, ceil(5.94)) = min(5, 6) = 5
    # 5th element in sorted [1,2,3,4,5] is 5.0
    assert threshold == 5.0, f"Expected threshold=5.0, got {threshold}"
    
    # Test with actual N-BaIoT REF values
    scores = np.random.randn(4500)
    threshold = compute_quantile_threshold(scores, alpha=0.01)
    sorted_scores = np.sort(scores)
    expected = sorted_scores[4455]  # 0-indexed
    assert abs(threshold - expected) < 1e-10, "Threshold doesn't match expected"
    
    print("Quantile baselines verification passed.")


if __name__ == "__main__":
    verify_quantile()
