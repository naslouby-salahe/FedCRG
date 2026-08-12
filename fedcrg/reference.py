"""
FedCRG Reference Threshold Module

Implements federation reference threshold construction per Section 5.1 of the
FedCRG Roadmap v2.0.

The reference threshold is a federation-wide operating point against which each
client is independently audited. It is NOT claimed to satisfy a per-client FPR
guarantee.

Key formulas:
    R = union_k R_k, with |R_k| identical across clients
    N_R = sum_k |R_k|
    q_ref = min(N_R, ceil((N_R + 1)(1 - alpha)))
    tau_ref = R_(q_ref), where R_(j) is the j-th ascending pooled score

For N-BaIoT primary: K=9, |R_k|=500, N_R=4500, q_ref=4456
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping, NamedTuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence


# =============================================================================
# LOCKED PROTOCOL CONSTANTS (Section 7-20, Appendix A)
# =============================================================================

class Alpha(float):
    """Target benign false-positive rate. LOCKED to 0.01."""

    def __new__(cls) -> "Alpha":
        return super().__new__(cls, 0.01)  # type: ignore[arg-type]


class Rho(float):
    """Relative practical tolerance around alpha. LOCKED to 0.50."""

    def __new__(cls) -> "Rho":
        return super().__new__(cls, 0.50)  # type: ignore[arg-type]


class A(float):
    """Lower acceptable FPR = max(0, alpha * (1 - rho)). DERIVED: 0.005."""

    def __new__(cls) -> "A":
        return super().__new__(cls, max(0.0, Alpha() * (1.0 - Rho())))  # type: ignore[arg-type]


class B(float):
    """Upper acceptable FPR = min(1, alpha * (1 + rho)). DERIVED: 0.015."""

    def __new__(cls) -> "B":
        return super().__new__(cls, min(1.0, Alpha() * (1.0 + Rho())))  # type: ignore[arg-type]


class GammaA(float):
    """Required Gate-A in-band assurance. LOCKED to 0.95."""

    def __new__(cls) -> "GammaA":
        return super().__new__(cls, 0.95)  # type: ignore[arg-type]


class GammaB(float):
    """Gate-B exact confidence level. LOCKED to 0.95."""

    def __new__(cls) -> "GammaB":
        return super().__new__(cls, 0.95)  # type: ignore[arg-type]


# Convenience aliases for primary values
PrimaryAlpha = Alpha
PrimaryRho = Rho
PrimaryA = A
PrimaryB = B
PrimaryGammaA = GammaA
PrimaryGammaB = GammaB


# =============================================================================
# REFERENCE THRESHOLD RESULT
# =============================================================================


class ReferenceThresholdResult(NamedTuple):
    """
    Result of building a federation reference threshold.

    Attributes:
        tau_ref: The reference threshold value (the q_ref-th order statistic)
        q_ref: The rank used (1-indexed in the sorted pooled scores)
        n_r: Total number of reference scores (sum of |R_k| across clients)
        n_clients: Number of clients contributing to the reference
        scores_per_client: Number of reference scores per client (should be equal)
        sorted_scores: The complete sorted pooled reference scores (for audit)
    """

    tau_ref: float
    q_ref: int
    n_r: int
    n_clients: int
    scores_per_client: int
    sorted_scores: npt.NDArray[np.float64]


# =============================================================================
# REFERENCE THRESHOLD CONSTRUCTION
# =============================================================================


def compute_q_ref(n_r: int, alpha: float = Alpha()) -> int:
    """
    Compute the reference rank q_ref.

    Formula: q_ref = min(N_R, ceil((N_R + 1)(1 - alpha)))

    For N-BaIoT primary: n_r=4500, alpha=0.01 => q_ref=4456

    Args:
        n_r: Total number of reference scores (N_R = sum_k |R_k|)
        alpha: Target FPR (default: Alpha() = 0.01)

    Returns:
        The 1-indexed rank for the reference threshold.
    """
    return min(n_r, math.ceil((n_r + 1) * (1.0 - alpha)))


def build_reference_threshold(
    reference_scores_by_client: Mapping[str, npt.NDArray[np.float64]],
    alpha: float = Alpha(),
) -> ReferenceThresholdResult:
    """
    Build the federation reference threshold from per-client reference scores.

    Normative implementation of Section 5.1:
        R = union_k R_k, with |R_k| identical across clients
        N_R = sum_k |R_k|
        q_ref = min(N_R, ceil((N_R + 1)(1 - alpha)))
        tau_ref = R_(q_ref)

    Args:
        reference_scores_by_client: Mapping from client_id to array of reference scores.
            Each array should contain |R_k| scores for that client.
        alpha: Target FPR (default: Alpha() = 0.01)

    Returns:
        ReferenceThresholdResult with tau_ref and metadata.

    Raises:
        ValueError: If client arrays have different lengths (violates |R_k| identical).
        ValueError: If any client has zero reference scores.
    """
    # Validate inputs
    if not reference_scores_by_client:
        raise ValueError("No reference scores provided")

    n_clients = len(reference_scores_by_client)
    scores_per_client = None

    all_scores: list[npt.NDArray[np.float64]] = []
    for client_id, scores in reference_scores_by_client.items():
        if len(scores) == 0:
            raise ValueError(f"Client {client_id} has zero reference scores")
        if scores_per_client is None:
            scores_per_client = len(scores)
        elif len(scores) != scores_per_client:
            raise ValueError(
                f"Client {client_id} has {len(scores)} reference scores, "
                f"expected {scores_per_client}"
            )
        all_scores.append(scores)

    # Concatenate all scores
    pooled_scores = np.concatenate(all_scores, dtype=np.float64)
    n_r = len(pooled_scores)

    # Sort in ascending order (for order statistics)
    sorted_scores = np.sort(pooled_scores)

    # Compute q_ref
    q_ref = compute_q_ref(n_r, alpha)

    # Get the reference threshold (q_ref is 1-indexed, so index q_ref-1)
    # Note: numpy uses 0-indexing, so R_(q_ref) = sorted_scores[q_ref - 1]
    tau_ref = float(sorted_scores[q_ref - 1])

    return ReferenceThresholdResult(
        tau_ref=tau_ref,
        q_ref=q_ref,
        n_r=n_r,
        n_clients=n_clients,
        scores_per_client=scores_per_client,  # type: ignore[arg-type]
        sorted_scores=sorted_scores,
    )


# =============================================================================
# N-BaIoT PRIMARY CONSTANTS
# =============================================================================

NBaiotClients = 9
"""Number of natural federated clients in N-BaIoT. LOCKED to 9."""

NBaiotReferencePerClient = 500
"""Number of reference scores per client in N-BaIoT. LOCKED to 500."""

NBaiotTotalReference = NBaiotClients * NBaiotReferencePerClient
"""Total reference scores for N-BaIoT: 9 * 500 = 4500. DERIVED."""

NBaiotQRef = compute_q_ref(NBaiotTotalReference, Alpha())
"""Reference rank for N-BaIoT: min(4500, ceil(4501 * 0.99)) = 4456. DERIVED."""

# Assert expected values from roadmap
assert NBaiotTotalReference == 4500, "N-BaIoT total reference must be 4500"
assert NBaiotQRef == 4456, "N-BaIoT q_ref must be 4456"


# =============================================================================
# DIAD CONSTANTS
# =============================================================================

DiadReferencePerClient = 300
"""Number of reference scores per client in DIAD. LOCKED to 300."""

# DiadTotalReference depends on number of eligible clients (K_D), which is DATA-DEPENDENT
# So we provide a function instead of a constant

def diad_q_ref(n_clients: int, alpha: float = Alpha()) -> tuple[int, int]:
    """
    Compute DIAD reference count and q_ref for given number of eligible clients.

    Args:
        n_clients: Number of eligible DIAD clients (K_D)
        alpha: Target FPR (default: Alpha() = 0.01)

    Returns:
        Tuple of (total_reference, q_ref)
    """
    total_reference = n_clients * DiadReferencePerClient
    q_ref = compute_q_ref(total_reference, alpha)
    return total_reference, q_ref


# =============================================================================
# GATE-B MINIMUM EVIDENCE
# =============================================================================


def compute_n_g_min(a: float, gamma_b: float = GammaB()) -> int:
    """
    Compute the minimum Gate-B sample size for given a and gamma_B.

    Formula from Section 5.3.2:
        n_{G,min}(a, gamma_B) = min{n >= 1: 1 - ((1 - gamma_B) / 2)^(1/n) < a}

    For primary contract (a=0.005, gamma_B=0.95): n_{G,min} = 736

    Args:
        a: Lower band limit (must be > 0 for bidirectional minimum)
        gamma_b: Confidence level

    Returns:
        Minimum sample size, or None if a=0 (one-sided by design)
    """
    if a <= 0:
        # When a=0, low-side mismatch is mathematically impossible
        # Return None to indicate one-sided band by design
        return 0  # Will be handled as special case

    delta = (1.0 - gamma_b) / 2.0
    n = 1
    while True:
        upper_bound = 1.0 - (delta ** (1.0 / n))
        if upper_bound < a:
            return n
        n += 1
        # Safety check to prevent infinite loop
        if n > 10_000:
            raise ValueError(
                f"Could not find n_G_min for a={a}, gamma_b={gamma_b}. "
                f"Current n={n}, upper_bound={upper_bound}"
            )


PrimaryNGMin = compute_n_g_min(PrimaryA(), PrimaryGammaB())
"""Primary Gate-B minimum: 736 for a=0.005, gamma_B=0.95. DERIVED."""

# Assert expected value from roadmap
assert PrimaryNGMin == 736, "Primary n_G_min must be 736"


# =============================================================================
# RUN-ID FORMATTING (Appendix B)
# =============================================================================


def format_run_id(
    dataset: str,
    detector: str,
    model_seed: int,
    cal_seed: int,
    policy: str,
    alpha: float = Alpha(),
    rho: float = Rho(),
    gamma_a: float = GammaA(),
    gamma_b: float = GammaB(),
) -> str:
    """
    Format a run ID according to Appendix B.2.

    Format: {dataset}__{detector}__ms{model_seed}__cs{cal_seed}__
            a{alpha_ppm}__r{rho_bp}__ga{gammaA_bp}__gb{gammaB_bp}__
            {policy}

    Where:
        alpha_ppm = round(alpha * 1_000_000)
        rho_bp = round(rho * 10_000)
        gamma_bp = round(gamma * 10_000)

    Args:
        dataset: Dataset identifier (e.g., "nbaiot", "diad")
        detector: Detector identifier (e.g., "ae", "deep_svdd")
        model_seed: Model seed
        cal_seed: Calibration seed
        alpha: Target FPR
        rho: Relative tolerance
        gamma_a: Gate-A assurance
        gamma_b: Gate-B confidence
        policy: Policy identifier

    Returns:
        Formatted run ID string
    """
    alpha_ppm = round(alpha * 1_000_000)
    rho_bp = round(rho * 10_000)
    gamma_a_bp = round(gamma_a * 10_000)
    gamma_b_bp = round(gamma_b * 10_000)

    return (
        f"{dataset}__{detector}__ms{model_seed}__cs{cal_seed}__"
        f"a{alpha_ppm}__r{rho_bp}__ga{gamma_a_bp}__gb{gamma_b_bp}__"
        f"{policy}"
    )


# Example from roadmap
_NBAIOT_EXAMPLE_RUN_ID = format_run_id(
    dataset="nbaiot",
    detector="ae",
    model_seed=11,
    cal_seed=1000,
    alpha=0.01,
    rho=0.5,
    gamma_a=0.95,
    gamma_b=0.95,
    policy="fedcrg",
)
assert _NBAIOT_EXAMPLE_RUN_ID == "nbaiot__ae__ms11__cs1000__a10000__r5000__ga9500__gb9500__fedcrg"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "Alpha",
    "Rho",
    "A",
    "B",
    "GammaA",
    "GammaB",
    "PrimaryAlpha",
    "PrimaryRho",
    "PrimaryA",
    "PrimaryB",
    "PrimaryGammaA",
    "PrimaryGammaB",
    # N-BaIoT constants
    "NBaiotClients",
    "NBaiotReferencePerClient",
    "NBaiotTotalReference",
    "NBaiotQRef",
    # DIAD constants
    "DiadReferencePerClient",
    "diad_q_ref",
    # Gate-B minimum
    "compute_n_g_min",
    "PrimaryNGMin",
    # Reference threshold
    "ReferenceThresholdResult",
    "compute_q_ref",
    "build_reference_threshold",
    # Run ID
    "format_run_id",
]
