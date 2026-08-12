"""
FedCRG Gate B Module - Independent Evidence of Reference Mismatch

Implements Gate B per Section 5.3 of the FedCRG Roadmap v2.0.

Gate B tests whether the federation reference threshold is materially
inappropriate for a client, using independent benign gate scores and the
exact Clopper-Pearson interval.

Core procedure:
    1. Count exceedances: x_k = sum_{g in G_k} 1[g > tau_ref]
    2. Compute two-sided 95% exact Clopper-Pearson interval [L_k, U_k]
       for p_ref,k = x_k / n_G
    3. LOW_MISMATCH if U_k < a
       HIGH_MISMATCH if L_k > b
       Otherwise NO_MATERIAL_MISMATCH_DEMONSTRATED

Clopper-Pearson formulas (Section 5.3.1):
    delta_B = 1 - gamma_B = 0.05
    
    L(x,n) = Beta^{-1}(delta_B/2; x, n-x+1) for x > 0, else 0
    U(x,n) = Beta^{-1}(1-delta_B/2; x+1, n-x) for x < n, else 1

Minimum evidence (Section 5.3.2):
    n_{G,min}(a, gamma_B) = min{n >= 1: 1 - ((1-gamma_B)/2)^(1/n) < a}
    For primary (a=0.005, gamma_B=0.95): n_{G,min} = 736

Diagnostic p-values (Section 5.3.1):
    p_low = Pr[X ~ Bin(n, a)][X <= x]
    p_high = Pr[X ~ Bin(n, b)][X >= x]

These are for auditing only; the normative decision is based on the
Clopper-Pearson interval rule.

Normative reference: Sections 5.3, 5.3.1, 5.3.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy import special
from scipy.stats import binom

from fedcrg.reference import A, B, GammaB, PrimaryA, PrimaryGammaB
from fedcrg.states import GateBMismatchState

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# GATE B RESULT
# =============================================================================


@dataclass(frozen=True, slots=True)
class GateBResult:
    """
    Result of Gate B computation for a single client.

    Attributes:
        n: Number of gate scores (|G_k|)
        x: Number of exceedances (sum of g > tau_ref)
        fpr_hat: Estimated reference FPR = x / n
        cp_lower: Lower bound of 95% exact Clopper-Pearson interval
        cp_upper: Upper bound of 95% exact Clopper-Pearson interval
        p_low: Diagnostic p-value for low-side test
        p_high: Diagnostic p-value for high-side test
        mismatch_state: The mismatch state (LOW, HIGH, or NONE)
        a: Lower band limit
        b: Upper band limit
        gamma_b: Confidence level
        n_g_min: Minimum required n for this (a, gamma_b)
        gate_scores: The gate scores (for audit)
    """

    n: int
    x: int
    fpr_hat: float
    cp_lower: float
    cp_upper: float
    p_low: float
    p_high: float
    mismatch_state: GateBMismatchState
    a: float
    b: float
    gamma_b: float
    n_g_min: int
    gate_scores: npt.NDArray[np.float64] | None = None


# =============================================================================
# CLOPPER-PEARSON CALCULATIONS
# =============================================================================


def compute_clopper_pearson_lower(x: int, n: int, delta_b: float = 0.025) -> float:
    """
    Compute the lower bound of the Clopper-Pearson interval.

    Formula from Section 5.3.1:
        L(x,n) = Beta^{-1}(delta_B/2; x, n-x+1) for x > 0, else 0

    where delta_B = (1 - gamma_B) / 2 = 0.025 for gamma_B=0.95

    Args:
        x: Number of successes (exceedances)
        n: Number of trials (gate scores)
        delta_b: Half of (1 - gamma_B), default 0.025 for 95% confidence

    Returns:
        Lower bound L(x,n)
    """
    if x == 0:
        return 0.0
    
    # L(x,n) = Beta^{-1}(delta_b; x, n-x+1)
    # scipy.special.betaincinv(a, b, y) computes the inverse regularized incomplete beta
    return float(special.betaincinv(x, n - x + 1, delta_b))


def compute_clopper_pearson_upper(x: int, n: int, delta_b: float = 0.025) -> float:
    """
    Compute the upper bound of the Clopper-Pearson interval.

    Formula from Section 5.3.1:
        U(x,n) = Beta^{-1}(1-delta_B/2; x+1, n-x) for x < n, else 1

    Args:
        x: Number of successes (exceedances)
        n: Number of trials (gate scores)
        delta_b: Half of (1 - gamma_B), default 0.025 for 95% confidence

    Returns:
        Upper bound U(x,n)
    """
    if x == n:
        return 1.0
    
    # U(x,n) = Beta^{-1}(1-delta_b; x+1, n-x)
    return float(special.betaincinv(x + 1, n - x, 1.0 - delta_b))


def compute_clopper_pearson_interval(
    x: int,
    n: int,
    gamma_b: float = GammaB(),
) -> tuple[float, float]:
    """
    Compute the exact two-sided Clopper-Pearson interval.

    Args:
        x: Number of exceedances
        n: Number of gate scores
        gamma_b: Confidence level (default: 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    delta_b = (1.0 - gamma_b) / 2.0
    lower = compute_clopper_pearson_lower(x, n, delta_b)
    upper = compute_clopper_pearson_upper(x, n, delta_b)
    return lower, upper


# =============================================================================
# DIAGNOSTIC P-VALUES
# =============================================================================


def compute_p_low(x: int, n: int, a: float) -> float:
    """
    Compute the diagnostic p-value for the low-side test.

    Formula from Section 5.3.1:
        p_low = Pr[X ~ Bin(n, a)][X <= x]

    This is the CDF of the binomial distribution at x.

    Args:
        x: Observed number of exceedances
        n: Number of gate scores
        a: Lower band limit (success probability under null)

    Returns:
        Left-tail p-value
    """
    return float(binom.cdf(x, n, a))


def compute_p_high(x: int, n: int, b: float) -> float:
    """
    Compute the diagnostic p-value for the high-side test.

    Formula from Section 5.3.1:
        p_high = Pr[X ~ Bin(n, b)][X >= x]

    This is the survival function (1 - CDF) of the binomial distribution.

    Args:
        x: Observed number of exceedances
        n: Number of gate scores
        b: Upper band limit (success probability under null)

    Returns:
        Right-tail p-value
    """
    return float(binom.sf(x - 1, n, b))  # sf is P[X > x], so P[X >= x] = P[X > x-1]


# =============================================================================
# MINIMUM EVIDENCE CALCULATION
# =============================================================================


def compute_n_g_min_dynamic(a: float, gamma_b: float = GammaB()) -> int:
    """
    Compute the minimum Gate-B sample size dynamically.

    Formula from Section 5.3.2:
        n_{G,min}(a, gamma_B) = min{n >= 1: 1 - ((1 - gamma_B) / 2)^(1/n) < a}

    For primary contract (a=0.005, gamma_B=0.95): n_{G,min} = 736

    Args:
        a: Lower band limit
        gamma_b: Confidence level

    Returns:
        Minimum sample size, or 0 if a <= 0 (one-sided by design)
    """
    if a <= 0:
        return 0
    
    delta = (1.0 - gamma_b) / 2.0
    n = 1
    while True:
        upper_bound = 1.0 - (delta ** (1.0 / n))
        if upper_bound < a:
            return n
        n += 1
        if n > 10_000:
            raise ValueError(f"Could not find n_G_min for a={a}, gamma_b={gamma_b}")


# =============================================================================
# MAIN GATE B COMPUTATION
# =============================================================================


def compute_gate_b(
    gate_scores: npt.NDArray[np.float64],
    tau_ref: float,
    a: float = PrimaryA(),
    b: float = B(),
    gamma_b: float = PrimaryGammaB(),
) -> GateBResult:
    """
    Compute Gate B result for a client's gate scores.

    Normative implementation of Section 5.3:
        1. x = count(g > tau_ref for g in G_k)
        2. [L, U] = exact two-sided Clopper-Pearson(x, |G_k|, gamma_B)
        3. LOW_MISMATCH if U < a
           HIGH_MISMATCH if L > b
           Otherwise NO_MATERIAL_MISMATCH_DEMONSTRATED
        4. Compute diagnostic p-values
        5. Compute n_{G,min}(a, gamma_B)

    Args:
        gate_scores: Array of n_G benign gate scores
        tau_ref: Federation reference threshold
        a: Lower band limit
        b: Upper band limit
        gamma_b: Confidence level

    Returns:
        GateBResult with all computed values.
    """
    n = len(gate_scores)
    
    if n == 0:
        raise ValueError("No gate scores provided")

    # Step 1: Count exceedances
    x = int(np.sum(gate_scores > tau_ref))
    
    # Step 2: Compute Clopper-Pearson interval
    cp_lower, cp_upper = compute_clopper_pearson_interval(x, n, gamma_b)
    
    # Step 3: Determine mismatch state
    if cp_upper < a:
        mismatch_state = GateBMismatchState.LOW_MISMATCH
    elif cp_lower > b:
        mismatch_state = GateBMismatchState.HIGH_MISMATCH
    else:
        mismatch_state = GateBMismatchState.NO_MATERIAL_MISMATCH_DEMONSTRATED
    
    # Step 4: Compute diagnostic p-values
    p_low = compute_p_low(x, n, a)
    p_high = compute_p_high(x, n, b)
    
    # Step 5: Compute minimum evidence
    n_g_min = compute_n_g_min_dynamic(a, gamma_b)
    
    # Compute estimated FPR
    fpr_hat = x / n if n > 0 else 0.0
    
    return GateBResult(
        n=n,
        x=x,
        fpr_hat=fpr_hat,
        cp_lower=cp_lower,
        cp_upper=cp_upper,
        p_low=p_low,
        p_high=p_high,
        mismatch_state=mismatch_state,
        a=a,
        b=b,
        gamma_b=gamma_b,
        n_g_min=n_g_min,
        gate_scores=gate_scores,
    )


# =============================================================================
# EXACT CUTOFFS FROM ROADMAP
# =============================================================================

# Expected cutoffs from Sections 380-387 and 736-746
_EXPECTED_CUTOFFS = {
    (736, 0.005, 0.95): {"low_max": 0, "high_min": 19},  # LOW if x=0, HIGH if x>=19
    (1000, 0.005, 0.95): {"low_max": 0, "high_min": 24},
    (1500, 0.005, 0.95): {"low_max": 2, "high_min": 33},
    (2000, 0.005, 0.95): {"low_max": 3, "high_min": 42},
    (3000, 0.005, 0.95): {"low_max": 7, "high_min": 59},
}


def get_expected_cutoffs(n: int, a: float = 0.005, gamma_b: float = 0.95) -> tuple[int, int] | None:
    """
    Get expected low/high cutoffs for given n, a, gamma_b.

    Args:
        n: Number of gate scores
        a: Lower band limit
        gamma_b: Confidence level

    Returns:
        Tuple of (low_max, high_min) or None if not in expected table.
    """
    key = (n, a, gamma_b)
    if key in _EXPECTED_CUTOFFS:
        return _EXPECTED_CUTOFFS[key]["low_max"], _EXPECTED_CUTOFFS[key]["high_min"]
    return None


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================


def verify_gate_b_exact_cutoffs(tolerance: float = 1e-10) -> bool:
    """
    Verify Gate B implementation against roadmap exact cutoffs.

    Per Section 1727-1753: exact cutoffs must match.

    Args:
        tolerance: Tolerance for interval comparisons (not used for integer cutoffs)

    Returns:
        True if all expected cutoffs match.
    """
    # Test the boundary cases from roadmap
    test_cases = [
        # (n, x, expected_state)
        (736, 0, GateBMismatchState.LOW_MISMATCH),
        (736, 1, GateBMismatchState.NO_MATERIAL_MISMATCH_DEMONSTRATED),
        (1000, 0, GateBMismatchState.LOW_MISMATCH),
        (1000, 1, GateBMismatchState.NO_MATERIAL_MISMATCH_DEMONSTRATED),
        (1500, 2, GateBMismatchState.LOW_MISMATCH),
        (1500, 3, GateBMismatchState.NO_MATERIAL_MISMATCH_DEMONSTRATED),
        (1500, 32, GateBMismatchState.NO_MATERIAL_MISMATCH_DEMONSTRATED),
        (1500, 33, GateBMismatchState.HIGH_MISMATCH),
        (3000, 7, GateBMismatchState.LOW_MISMATCH),
        (3000, 8, GateBMismatchState.NO_MATERIAL_MISMATCH_DEMONSTRATED),
        (3000, 58, GateBMismatchState.NO_MATERIAL_MISMATCH_DEMONSTRATED),
        (3000, 59, GateBMismatchState.HIGH_MISMATCH),
    ]
    
    a = PrimaryA()
    b = B()
    gamma_b = PrimaryGammaB()
    
    for n, x, expected_state in test_cases:
        # Create dummy scores: x scores > tau_ref, n-x scores <= tau_ref
        # Use tau_ref = 0 for simplicity, with x positive scores and n-x negative scores
        gate_scores = np.array([1.0] * x + [-1.0] * (n - x), dtype=np.float64)
        
        result = compute_gate_b(gate_scores, tau_ref=0.0, a=a, b=b, gamma_b=gamma_b)
        
        if result.mismatch_state != expected_state:
            print(
                f"FAIL: n={n}, x={x}, expected {expected_state}, "
                f"got {result.mismatch_state}, CP=[{result.cp_lower:.6f}, {result.cp_upper:.6f}]"
            )
            return False
    
    print("All Gate B exact cutoffs verified")
    return True


# =============================================================================
# POWER CALCULATION (Section 19.50-19.66)
# =============================================================================


def compute_gate_b_power(
    true_fpr: float,
    n_g: int,
    a: float = PrimaryA(),
    b: float = B(),
    gamma_b: float = PrimaryGammaB(),
) -> float:
    """
    Compute the probability that Gate B declares mismatch for a given true FPR.

    This computes the probability over the binomial distribution Bin(n_g, true_fpr)
    that the Clopper-Pearson interval falls entirely below a (LOW) or entirely
    above b (HIGH).

    Args:
        true_fpr: True reference FPR
        n_g: Number of gate scores
        a: Lower band limit
        b: Upper band limit
        gamma_b: Confidence level

    Returns:
        Probability of mismatch declaration (LOW or HIGH)
    """
    delta_b = (1.0 - gamma_b) / 2.0
    
    # We need to sum over all x where CP interval is entirely below a or above b
    # For LOW: U(x, n_g) < a  =>  Beta^{-1}(1-delta_b; x+1, n_g-x) < a
    # For HIGH: L(x, n_g) > b  =>  Beta^{-1}(delta_b; x, n_g-x+1) > b
    
    mismatch_count = 0
    total_trials = 100_000  # Monte Carlo trials for approximation
    
    for _ in range(total_trials):
        # Sample x from Bin(n_g, true_fpr)
        x = int(np.random.binomial(n_g, true_fpr))
        
        # Compute CP interval for this x
        cp_lower, cp_upper = compute_clopper_pearson_interval(x, n_g, gamma_b)
        
        # Check mismatch
        if cp_upper < a or cp_lower > b:
            mismatch_count += 1
    
    return mismatch_count / total_trials


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "GateBResult",
    # Functions
    "compute_clopper_pearson_lower",
    "compute_clopper_pearson_upper",
    "compute_clopper_pearson_interval",
    "compute_p_low",
    "compute_p_high",
    "compute_n_g_min_dynamic",
    "compute_gate_b",
    "get_expected_cutoffs",
    "verify_gate_b_exact_cutoffs",
    "compute_gate_b_power",
]
