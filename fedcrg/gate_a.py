"""
FedCRG Gate A Module - Local Operating-Band Readiness

Implements Gate A per Section 5.2 of the FedCRG Roadmap v2.0.

Gate A determines whether a client has enough independent benign calibration
data to construct a local threshold whose future benign FPR falls inside the
operating band [a, b] with probability at least gamma_A.

Core formula:
    Let C_k = {c_1, ..., c_n} be n i.i.d. continuous benign scores, sorted.
    For candidate rank r, tau_r = c_(r) (r-th order statistic).
    
    Under the i.i.d.-continuous model:
        P_r = Pr[a <= FPR(tau_r) <= b]
            = I_b(n+1-r, r) - I_a(n+1-r, r)
    
    where I_z(.,.) is the regularized incomplete beta function.
    
    r* = argmax_r P_r (ties: larger r wins)
    Gate A is READY iff max_r P_r >= gamma_A
    tau_local = sorted(C_k)[r*-1]  (0-indexed: index r*-1)

For primary N-BaIoT (n=2000, alpha=0.01, rho=0.5, gamma_A=0.95):
    r* = 1982, P_r = 0.9805279151

Precomputation requirement (Section 14.5):
    For fixed (n, a, b, gamma_A), r* and P_r are determined BEFORE observing scores.
    Runtime MUST read precomputed rank, MUST NOT optimize using observed scores.

Numerical requirement (Section 347):
    Beta-CDF calculations MUST use float64, absolute error <= 1e-10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy import special

from fedcrg.reference import A, B, GammaA, PrimaryAlpha, PrimaryRho

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# PRECOMPUTED GATE A TABLE
# =============================================================================


@dataclass(frozen=True, slots=True)
class GateATableEntry:
    """
    A single entry in the Gate A precomputation table.

    Attributes:
        n: Number of calibration samples
        rank_r: The optimal rank r* (1-indexed)
        coverage_probability: P_r at rank r*
        ready: Whether max_r P_r >= gamma_A
        alpha: Target FPR used
        rho: Tolerance used
        a: Lower band = max(0, alpha*(1-rho))
        b: Upper band = min(1, alpha*(1+rho))
        gamma_a: Gate-A assurance level
    """

    n: int
    rank_r: int
    coverage_probability: float
    ready: bool
    alpha: float
    rho: float
    a: float
    b: float
    gamma_a: float


class GateATable:
    """
    Precomputed Gate A table for lookup.

    This table contains r* and P_r for various (n, alpha, rho, gamma_A) combinations.
    Per Section 14.5 and 338-341: runtime code MUST read precomputed rank and
    MUST NOT optimize rank using observed client scores.

    The table is built lazily and cached for performance.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[int, float, float, float], GateATableEntry] = {}

    def get(self, n: int, alpha: float, rho: float, gamma_a: float) -> GateATableEntry:
        """
        Get or compute a Gate A table entry.

        Args:
            n: Number of calibration samples
            alpha: Target FPR
            rho: Relative tolerance
            gamma_a: Gate-A assurance level

        Returns:
            The Gate A table entry for these parameters.
        """
        key = (n, alpha, rho, gamma_a)
        if key not in self._entries:
            self._entries[key] = self._compute_entry(n, alpha, rho, gamma_a)
        return self._entries[key]

    @staticmethod
    def _compute_entry(n: int, alpha: float, rho: float, gamma_a: float) -> GateATableEntry:
        """
        Compute a single Gate A table entry.

        Args:
            n: Number of calibration samples
            alpha: Target FPR
            rho: Relative tolerance
            gamma_a: Gate-A assurance level

        Returns:
            The computed Gate A table entry.
        """
        a = max(0.0, alpha * (1.0 - rho))
        b = min(1.0, alpha * (1.0 + rho))

        # Find r* = argmax_r P_r, with ties broken to larger r
        # P_r = I_b(n+1-r, r) - I_a(n+1-r, r)
        # We need to search r from 1 to n
        best_r = 0
        best_p = -1.0

        for r in range(1, n + 1):
            p_r = _compute_p_r(r, n, a, b)
            if p_r > best_p or (math.isclose(p_r, best_p) and r > best_r):
                best_p = p_r
                best_r = r

        ready = best_p >= gamma_a

        return GateATableEntry(
            n=n,
            rank_r=best_r,
            coverage_probability=best_p,
            ready=ready,
            alpha=alpha,
            rho=rho,
            a=a,
            b=b,
            gamma_a=gamma_a,
        )


# Global precomputation table instance
_gate_a_table = GateATable()


# =============================================================================
# GATE A RESULT
# =============================================================================


@dataclass(frozen=True, slots=True)
class GateAResult:
    """
    Result of Gate A computation for a single client.

    Attributes:
        n: Number of calibration samples
        rank: The optimal rank r* (1-indexed)
        coverage_probability: P_r at rank r*
        ready: Whether Gate A is READY (P_r >= gamma_A)
        tau_local: The local threshold value (if ready)
        tie_count: Multiplicity of tau_local in calibration scores
        a: Lower band limit
        b: Upper band limit
        gamma_a: Gate-A assurance level
        sorted_calibration_scores: Sorted calibration scores (for audit)
    """

    n: int
    rank: int
    coverage_probability: float
    ready: bool
    tau_local: float | None
    tie_count: int
    a: float
    b: float
    gamma_a: float
    sorted_calibration_scores: npt.NDArray[np.float64] | None = None


# =============================================================================
# CORE COMPUTATION FUNCTIONS
# =============================================================================


def _compute_p_r(r: int, n: int, a: float, b: float) -> float:
    """
    Compute P_r = I_b(n+1-r, r) - I_a(n+1-r, r).

    This is the exact probability that the future benign FPR falls inside
    [a, b] when using the r-th order statistic as threshold.

    Under the i.i.d.-continuous model:
        P_FP(C_{k,(r)}) ~ Beta(n+1-r, r)
        P_r = Pr[a <= P_FP <= b] = I_b(n+1-r, r) - I_a(n+1-r, r)

    Args:
        r: Candidate rank (1-indexed)
        n: Number of calibration samples
        a: Lower band limit
        b: Upper band limit

    Returns:
        P_r for this rank.
    """
    # Regularized incomplete beta function: I_x(a, b) = betainc(a, b, x)
    # scipy.special.betainc(a, b, x) computes I_x(a, b)
    # We need I_b(n+1-r, r) - I_a(n+1-r, r)
    
    lower_tail = special.betainc(n + 1 - r, r, a)
    upper_tail = special.betainc(n + 1 - r, r, b)
    
    return upper_tail - lower_tail


def compute_gate_a(
    calibration_scores: npt.NDArray[np.float64],
    alpha: float = PrimaryAlpha(),
    rho: float = PrimaryRho(),
    gamma_a: float = GammaA(),
    precomputed_table: GateATable | None = None,
) -> GateAResult:
    """
    Compute Gate A result for a client's calibration scores.

    Normative implementation of Section 5.2:
        1. Sort calibration scores: c_(1) <= ... <= c_(n)
        2. For fixed (n, a, b, gamma_A), look up precomputed r* and P_r
           (MUST NOT optimize r using observed scores)
        3. Gate A is READY iff P_r >= gamma_A
        4. tau_local = c_(r*) (the r*-th order statistic)
        5. Check tie_count at tau_local

    Args:
        calibration_scores: Array of n benign calibration scores
        alpha: Target FPR
        rho: Relative tolerance
        gamma_a: Gate-A assurance level
        precomputed_table: Optional precomputed table (defaults to global)

    Returns:
        GateAResult with all computed values.
    """
    n = len(calibration_scores)
    
    if n == 0:
        raise ValueError("No calibration scores provided")

    # Compute band limits
    a = max(0.0, alpha * (1.0 - rho))
    b = min(1.0, alpha * (1.0 + rho))

    # Get precomputed entry
    table = precomputed_table if precomputed_table is not None else _gate_a_table
    entry = table.get(n, alpha, rho, gamma_a)

    # Sort scores in ascending order
    sorted_scores = np.sort(calibration_scores, kind="stable")

    # Determine tau_local and tie_count
    tau_local: float | None = None
    tie_count = 0

    if entry.ready:
        # r* is 1-indexed, so index is r*-1 in 0-indexed array
        rank_index = entry.rank_r - 1
        tau_local = float(sorted_scores[rank_index])
        
        # Count multiplicity at tau_local
        # Find all indices where sorted_scores == tau_local
        tie_count = int(np.sum(sorted_scores == tau_local))

    return GateAResult(
        n=n,
        rank=entry.rank_r,
        coverage_probability=entry.coverage_probability,
        ready=entry.ready,
        tau_local=tau_local,
        tie_count=tie_count,
        a=a,
        b=b,
        gamma_a=gamma_a,
        sorted_calibration_scores=sorted_scores,
    )


# =============================================================================
# PRECOMPUTATION FOR KNOWN VALUES
# =============================================================================


def precompute_primary_gate_a_table() -> dict[int, GateATableEntry]:
    """
    Precompute Gate A table entries for primary contract and known n values.

    Primary contract: alpha=0.01, rho=0.5, gamma_A=0.95
    Known n values from roadmap: 1415, 1416, 1500, 2000, 3000, etc.

    Returns:
        Dictionary mapping n to GateATableEntry for primary contract.
    """
    alpha = PrimaryAlpha()
    rho = PrimaryRho()
    gamma_a = GammaA()
    
    result = {}
    
    # Known n values from roadmap
    known_n_values = [
        500, 694, 1000, 1400, 1415, 1416, 1417, 1500, 2000, 2435, 2861, 3000,
        5722, 5970, 149, 270, 2810, 3341, 3971, 4430, 4470,
    ]
    
    for n in known_n_values:
        entry = _gate_a_table.get(n, alpha, rho, gamma_a)
        result[n] = entry
    
    return result


# =============================================================================
# EXACT VALUES FROM ROADMAP (for verification)
# =============================================================================

# Expected values from Section 349-367, 358-360, 462-468, G.1
# Using the more precise values from Appendix G.1 where available
_EXPECTED_VALUES = {
    (1415, 0.01, 0.5, 0.95): {"rank": 1403, "p": 0.9499884311, "ready": False},
    (1416, 0.01, 0.5, 0.95): {"rank": 1404, "p": 0.9500045311, "ready": True},
    (1500, 0.01, 0.5, 0.95): {"rank": 1487, "p": 0.9573928914, "ready": True},
    (2000, 0.01, 0.5, 0.95): {"rank": 1982, "p": 0.9805279151, "ready": True},
    (1000, 0.01, 0.5, 0.90): {"rank": 991, "p": 0.9001415746, "ready": True},
    (2435, 0.01, 0.5, 0.99): {"rank": 2413, "p": 0.9900229803, "ready": True},
}


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================


def verify_gate_a_exact_values(tolerance: float = 1e-10) -> bool:
    """
    Verify Gate A implementation against roadmap exact values.

    Per Section 347: absolute error against reference values must be <= 1e-10.

    Args:
        tolerance: Maximum allowed absolute error

    Returns:
        True if all expected values match within tolerance.
    """
    alpha = 0.01
    rho = 0.5
    gamma_a = 0.95
    
    for (n, a_exp, r_exp, ga_exp), expected in _EXPECTED_VALUES.items():
        entry = _gate_a_table.get(n, a_exp, r_exp, ga_exp)
        
        if entry.rank_r != expected["rank"]:
            print(f"FAIL: n={n}, expected rank={expected['rank']}, got {entry.rank_r}")
            return False
        
        if abs(entry.coverage_probability - expected["p"]) > tolerance:
            print(
                f"FAIL: n={n}, expected p={expected['p']}, "
                f"got {entry.coverage_probability}, diff={abs(entry.coverage_probability - expected['p'])}"
            )
            return False
        
        if entry.ready != expected["ready"]:
            print(f"FAIL: n={n}, expected ready={expected['ready']}, got {entry.ready}")
            return False
    
    print("All Gate A exact values verified within tolerance")
    return True


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "GateATableEntry",
    "GateATable",
    "GateAResult",
    # Functions
    "compute_gate_a",
    "precompute_primary_gate_a_table",
    "verify_gate_a_exact_values",
    # Module-level instances
    "_gate_a_table",
]
