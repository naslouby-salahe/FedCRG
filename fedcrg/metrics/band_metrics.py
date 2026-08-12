"""
FedCRG Band Metrics Module

Implements band-related metrics per Section 10 of the FedCRG Roadmap v2.0.

Key metrics:
- BandError_k = max{a - FPR_k, 0, FPR_k - b}
- MEBE = (1/K) * sum_k(BandError_k)
- HighExcess = max{0, max_k(FPR_k) - b}
- BandViolationRate = (1/K) * sum_k[I(FPR_k < a or FPR_k > b)]
- MAFE = (1/K) * sum_k|FPR_k - alpha|

Normative reference: Section 10, Equations 1331-1355
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from fedcrg.reference import A, B, Alpha

if TYPE_CHECKING:
    pass


# =============================================================================
# BAND METRIC DATACLASSES
# =============================================================================


@dataclass(frozen=True, slots=True)
class BandError:
    """
    Per-client band error metric.

    BandError_k = max{a - FPR_k, 0, FPR_k - b}

    This measures how far the client's FPR is from the acceptable band [a, b].

    Attributes:
        client_id: Client identifier
        fpr: Actual false positive rate for this client
        band_error: The band error value
        a: Lower band limit
        b: Upper band limit
        below_band: True if FPR < a
        above_band: True if FPR > b
        in_band: True if a <= FPR <= b
    """

    client_id: str
    fpr: float
    band_error: float
    a: float
    b: float
    below_band: bool = False
    above_band: bool = False
    in_band: bool = False

    def __post_init__(self):
        """Validate and set flags."""
        if self.fpr < self.a:
            object.__setattr__(self, "below_band", True)
        elif self.fpr > self.b:
            object.__setattr__(self, "above_band", True)
        else:
            object.__setattr__(self, "in_band", True)


@dataclass(frozen=True, slots=True)
class MEBEResult:
    """
    Mean Excess Band Error (MEBE) result.

    MEBE = (1/K) * sum_{k=1}^K BandError_k

    Lower is better. This is a primary reliability metric.

    Attributes:
        mebe: Mean Excess Band Error value
        n_clients: Number of clients
        client_band_errors: Dictionary mapping client_id to BandError
        a: Lower band limit
        b: Upper band limit
    """

    mebe: float
    n_clients: int
    client_band_errors: Dict[str, BandError]
    a: float
    b: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mebe": self.mebe,
            "n_clients": self.n_clients,
            "client_band_errors": {
                cid: {
                    "fpr": be.fpr,
                    "band_error": be.band_error,
                    "below_band": be.below_band,
                    "above_band": be.above_band,
                    "in_band": be.in_band,
                }
                for cid, be in self.client_band_errors.items()
            },
            "a": self.a,
            "b": self.b,
        }


@dataclass(frozen=True, slots=True)
class HighExcessResult:
    """
    HighExcess result.

    HighExcess = max{0, max_k(FPR_k) - b}

    This measures the worst-client excess above the upper band.
    Lower is better. This is a primary safety metric.

    Attributes:
        high_excess: HighExcess value
        max_fpr: Maximum FPR across all clients
        max_fpr_client: Client with maximum FPR
        b: Upper band limit
    """

    high_excess: float
    max_fpr: float
    max_fpr_client: str
    b: float


@dataclass(frozen=True, slots=True)
class BandViolationRateResult:
    """
    Band Violation Rate result.

    BandViolationRate = (1/K) * sum_{k=1}^K I(FPR_k < a or FPR_k > b)

    This is the fraction of clients with FPR outside the acceptable band.
    Lower is better. This is a secondary reliability metric.

    Attributes:
        band_violation_rate: Band violation rate value
        n_clients: Number of clients
        n_violations: Number of clients violating the band
        violating_clients: List of client IDs with band violations
        a: Lower band limit
        b: Upper band limit
    """

    band_violation_rate: float
    n_clients: int
    n_violations: int
    violating_clients: List[str]
    a: float
    b: float


@dataclass(frozen=True, slots=True)
class MAFEResult:
    """
    Mean Absolute FPR Error (MAFE) result.

    MAFE = (1/K) * sum_{k=1}^K |FPR_k - alpha|

    This is a secondary reliability metric.

    Attributes:
        mafe: Mean Absolute FPR Error value
        n_clients: Number of clients
        client_fpr_errors: Dictionary mapping client_id to |FPR_k - alpha|
        alpha: Target FPR
    """

    mafe: float
    n_clients: int
    client_fpr_errors: Dict[str, float]
    alpha: float


# =============================================================================
# COMPUTATION FUNCTIONS
# =============================================================================


def compute_band_error(
    fpr: float,
    a: float = A(),
    b: float = B(),
) -> float:
    """
    Compute per-client band error.

    BandError_k = max{a - FPR_k, 0, FPR_k - b}

    Args:
        fpr: False positive rate for this client
        a: Lower band limit (default: A() = 0.005)
        b: Upper band limit (default: B() = 0.015)

    Returns:
        Band error value (non-negative)

    Normative reference: Equation 1334-1336
    """
    return max(a - fpr, 0.0, fpr - b)


def compute_mebe(
    fprs: Dict[str, float],
    a: float = A(),
    b: float = B(),
) -> MEBEResult:
    """
    Compute Mean Excess Band Error (MEBE).

    MEBE = (1/K) * sum_{k=1}^K BandError_k

    Args:
        fprs: Dictionary mapping client_id to FPR value
        a: Lower band limit (default: A() = 0.005)
        b: Upper band limit (default: B() = 0.015)

    Returns:
        MEBEResult with all computed values

    Normative reference: Equation 1341-1342
    """
    n_clients = len(fprs)
    if n_clients == 0:
        raise ValueError("No clients provided")

    client_band_errors = {}
    total_band_error = 0.0

    for client_id, fpr in fprs.items():
        band_error = compute_band_error(fpr, a, b)
        client_band_errors[client_id] = BandError(
            client_id=client_id,
            fpr=fpr,
            band_error=band_error,
            a=a,
            b=b,
        )
        total_band_error += band_error

    mebe = total_band_error / n_clients

    return MEBEResult(
        mebe=mebe,
        n_clients=n_clients,
        client_band_errors=client_band_errors,
        a=a,
        b=b,
    )


def compute_high_excess(
    fprs: Dict[str, float],
    b: float = B(),
) -> HighExcessResult:
    """
    Compute HighExcess.

    HighExcess = max{0, max_k(FPR_k) - b}

    Args:
        fprs: Dictionary mapping client_id to FPR value
        b: Upper band limit (default: B() = 0.015)

    Returns:
        HighExcessResult with all computed values

    Normative reference: Equation 1345
    """
    if not fprs:
        raise ValueError("No clients provided")

    max_fpr = max(fprs.values())
    max_fpr_client = max(fprs, key=fprs.get)
    high_excess = max(0.0, max_fpr - b)

    return HighExcessResult(
        high_excess=high_excess,
        max_fpr=max_fpr,
        max_fpr_client=max_fpr_client,
        b=b,
    )


def compute_band_violation_rate(
    fprs: Dict[str, float],
    a: float = A(),
    b: float = B(),
) -> BandViolationRateResult:
    """
    Compute Band Violation Rate.

    BandViolationRate = (1/K) * sum_{k=1}^K I(FPR_k < a or FPR_k > b)

    Args:
        fprs: Dictionary mapping client_id to FPR value
        a: Lower band limit (default: A() = 0.005)
        b: Upper band limit (default: B() = 0.015)

    Returns:
        BandViolationRateResult with all computed values

    Normative reference: Equation 1349-1351
    """
    n_clients = len(fprs)
    if n_clients == 0:
        raise ValueError("No clients provided")

    violating_clients = []
    n_violations = 0

    for client_id, fpr in fprs.items():
        if fpr < a or fpr > b:
            violating_clients.append(client_id)
            n_violations += 1

    band_violation_rate = n_violations / n_clients

    return BandViolationRateResult(
        band_violation_rate=band_violation_rate,
        n_clients=n_clients,
        n_violations=n_violations,
        violating_clients=violating_clients,
        a=a,
        b=b,
    )


def compute_mafe(
    fprs: Dict[str, float],
    alpha: float = Alpha(),
) -> MAFEResult:
    """
    Compute Mean Absolute FPR Error (MAFE).

    MAFE = (1/K) * sum_{k=1}^K |FPR_k - alpha|

    Args:
        fprs: Dictionary mapping client_id to FPR value
        alpha: Target FPR (default: Alpha() = 0.01)

    Returns:
        MAFEResult with all computed values

    Normative reference: Equation 1354-1355
    """
    n_clients = len(fprs)
    if n_clients == 0:
        raise ValueError("No clients provided")

    client_fpr_errors = {}
    total_error = 0.0

    for client_id, fpr in fprs.items():
        error = abs(fpr - alpha)
        client_fpr_errors[client_id] = error
        total_error += error

    mafe = total_error / n_clients

    return MAFEResult(
        mafe=mafe,
        n_clients=n_clients,
        client_fpr_errors=client_fpr_errors,
        alpha=alpha,
    )


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================


def verify_band_metrics() -> bool:
    """
    Verify band metrics implementation.

    Tests:
    - BandError computation
    - MEBE computation
    - HighExcess computation
    - BandViolationRate computation
    - MAFE computation
    - Edge cases (FPR in band, below band, above band)
    """
    # Test data
    fprs = {
        "client_01": 0.004,   # Below band (a=0.005, b=0.015)
        "client_02": 0.010,   # In band
        "client_03": 0.020,   # Above band
        "client_04": 0.005,   # At lower bound
        "client_05": 0.015,   # At upper bound
    }

    a = 0.005
    b = 0.015

    # Test BandError
    be1 = compute_band_error(0.004, a, b)  # max{0.005-0.004, 0, 0.004-0.015} = 0.001
    assert abs(be1 - 0.001) < 1e-10, f"Expected 0.001, got {be1}"

    be2 = compute_band_error(0.010, a, b)  # max{0.005-0.010, 0, 0.010-0.015} = 0.0
    assert abs(be2 - 0.0) < 1e-10, f"Expected 0.0, got {be2}"

    be3 = compute_band_error(0.020, a, b)  # max{0.005-0.020, 0, 0.020-0.015} = 0.005
    assert abs(be3 - 0.005) < 1e-10, f"Expected 0.005, got {be3}"

    # Test MEBE
    mebe_result = compute_mebe(fprs, a, b)
    expected_mebe = (0.001 + 0.0 + 0.005 + 0.0 + 0.0) / 5  # = 0.0012
    assert abs(mebe_result.mebe - 0.0012) < 1e-10, f"Expected 0.0012, got {mebe_result.mebe}"

    # Test HighExcess
    he_result = compute_high_excess(fprs, b)
    assert abs(he_result.max_fpr - 0.020) < 1e-10
    assert he_result.max_fpr_client == "client_03"
    assert abs(he_result.high_excess - 0.005) < 1e-10

    # Test BandViolationRate
    bvr_result = compute_band_violation_rate(fprs, a, b)
    # client_01 (0.004 < 0.005) and client_03 (0.020 > 0.015) violate
    assert bvr_result.n_violations == 2
    assert abs(bvr_result.band_violation_rate - 0.4) < 1e-10

    # Test MAFE
    mafe_result = compute_mafe(fprs, alpha=0.01)
    # |0.004-0.01|=0.006, |0.010-0.01|=0.0, |0.020-0.01|=0.01, |0.005-0.01|=0.005, |0.015-0.01|=0.005
    # sum = 0.006 + 0.0 + 0.01 + 0.005 + 0.005 = 0.026
    # mean = 0.026 / 5 = 0.0052
    expected_mafe = 0.026 / 5
    assert abs(mafe_result.mafe - expected_mafe) < 1e-10, f"Expected {expected_mafe}, got {mafe_result.mafe}"

    print("Band metrics verification passed.")
    return True


if __name__ == "__main__":
    verify_band_metrics()
