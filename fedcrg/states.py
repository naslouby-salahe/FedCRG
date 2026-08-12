"""
FedCRG State Machine Module

Implements the FedCRG decision states and state machine per Section 5.4 of the
FedCRG Roadmap v2.0.

The FedCRG state machine combines Gate A (local readiness) and Gate B (reference
mismatch) outcomes to determine the appropriate threshold deployment state.

Five deployment states:
1. NO_MATERIAL_MISMATCH_DEMONSTRATED - Use tau_ref (Gate B inconclusive)
2. LOCAL_PERSONALIZE - Use tau_local (Gate B mismatch + Gate A ready + no tie)
3. CALIBRATION_DEFICIT - Use tau_ref temporarily (Gate B mismatch + Gate A not ready)
4. GATE_B_INSUFFICIENT - Use tau_ref temporarily (n_G < n_{G,min})
5. CALIBRATION_ASSUMPTION_VIOLATION - Use tau_ref temporarily (tie at threshold)

Normative reference: Section 5.4, Tables in 458-474
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

from fedcrg.reference import A, B, GammaA, GammaB, PrimaryNGMin, compute_n_g_min

if TYPE_CHECKING:
    from fedcrg.gate_a import GateAResult
    from fedcrg.gate_b import GateBResult
    from fedcrg.reference import ReferenceThresholdResult


# =============================================================================
# GATE A MISMATCH STATES (for internal use)
# =============================================================================


class GateAMismatchState(Enum):
    """
    Gate A outcome states.

    Gate A determines whether the client has enough local calibration data
    and can construct a threshold with sufficient in-band probability.
    """

    NOT_READY = auto()
    READY = auto()


# =============================================================================
# GATE B MISMATCH STATES
# =============================================================================


class GateBMismatchState(Enum):
    """
    Gate B outcome states.

    Gate B provides independent evidence that the reference threshold is
    materially inappropriate for the client.
    """

    LOW_MISMATCH = auto()
    HIGH_MISMATCH = auto()
    NO_MATERIAL_MISMATCH_DEMONSTRATED = auto()


# =============================================================================
# FEDCRG DEPLOYMENT STATES
# =============================================================================


class FedCRGState(Enum):
    """
    FedCRG deployment states per Section 5.4.

    These are the five normative deployment states that determine which
    threshold is used for each client.

    Attributes:
        use_tau_local: Whether to use the client-specific local threshold
        use_tau_ref: Whether to use the federation reference threshold
        description: Human-readable description of the state
    """

    NO_MATERIAL_MISMATCH_DEMONSTRATED = (
        False,
        True,
        "Reference retained; not called certified or equivalent",
    )
    LOCAL_PERSONALIZE = (
        True,
        False,
        "Only state counted as admitted personalization",
    )
    CALIBRATION_DEFICIT = (
        False,
        True,
        "Mismatch demonstrated but local certified replacement unavailable",
    )
    GATE_B_INSUFFICIENT = (
        False,
        True,
        "Reference status not evaluable; collect more G_k",
    )
    CALIBRATION_ASSUMPTION_VIOLATION = (
        False,
        True,
        "Observed tie violates continuity model; local deployment blocked",
    )

    def __new__(
        cls,
        use_tau_local: bool,
        use_tau_ref: bool,
        description: str,
    ) -> "FedCRGState":
        obj = object.__new__(cls)
        obj._value_ = auto().value
        obj.use_tau_local = use_tau_local
        obj.use_tau_ref = use_tau_ref
        obj.description = description
        return obj

    @property
    def selected_threshold(self) -> str:
        """Return which threshold is selected for this state."""
        if self.use_tau_local:
            return "tau_local"
        else:
            return "tau_ref"


# =============================================================================
# REASON CODES
# =============================================================================


class ReasonCode(Enum):
    """
    Machine-readable reason codes for FedCRG decisions.

    These codes provide detailed information about why a particular state
    was selected, useful for auditing and analysis.
    """

    # Gate A reasons
    GATE_A_NOT_READY = "GATE_A_NOT_READY"
    GATE_A_READY = "GATE_A_READY"
    GATE_A_TIE_VIOLATION = "CALIBRATION_ASSUMPTION_VIOLATION"

    # Gate B reasons
    GATE_B_LOW_MISMATCH = "LOW_MISMATCH"
    GATE_B_HIGH_MISMATCH = "HIGH_MISMATCH"
    GATE_B_NO_MISMATCH = "NO_MATERIAL_MISMATCH_DEMONSTRATED"
    GATE_B_INSUFFICIENT_EVIDENCE = "GATE_B_INSUFFICIENT"

    # Combined reasons
    LOCAL_PERSONALIZATION_ADMITTED = "LOCAL_PERSONALIZE"
    CALIBRATION_DEFICIT_STATE = "CALIBRATION_DEFICIT"


# =============================================================================
# FEDCRG DECISION RESULT
# =============================================================================


class FedCRGDecision(NamedTuple):
    """
    Complete FedCRG decision result.

    Contains all information needed to audit and reproduce the decision.

    Attributes:
        state: The deployment state
        selected_threshold: The threshold value to use
        selected_source: Which threshold source ("reference" or "local")
        tie_count: Multiplicity of the selected threshold (0 if not applicable)
        reason_code: Machine-readable reason for the decision
        gate_a_result: Full Gate A result (if computed)
        gate_b_result: Full Gate B result (if computed)
        reference_result: Reference threshold result (if available)
    """

    state: FedCRGState
    selected_threshold: float
    selected_source: str  # "reference" or "local"
    tie_count: int
    reason_code: ReasonCode
    gate_a_result: "GateAResult | None" = None
    gate_b_result: "GateBResult | None" = None
    reference_result: "ReferenceThresholdResult | None" = None


# =============================================================================
# DECISION FUNCTION
# =============================================================================


def decide_fedcrg(
    reference: "ReferenceThresholdResult",
    gate_a: "GateAResult",
    gate_b: "GateBResult",
) -> FedCRGDecision:
    """
    Implement the FedCRG state machine decision per Section 5.4 pseudocode.

    Normative implementation following the pseudocode in Section 5.4:
    
    1. If Gate B does not establish mismatch (NO_MATERIAL_MISMATCH_DEMONSTRATED):
       => NO_MATERIAL_MISMATCH_DEMONSTRATED, threshold=tau_ref
    
    2. If Gate B establishes mismatch:
       a. If n_G < n_{G,min}: => GATE_B_INSUFFICIENT, threshold=tau_ref
       b. If Gate A is not READY: => CALIBRATION_DEFICIT, threshold=tau_ref
       c. If Gate A is READY but selected threshold has tie_count > 1:
          => CALIBRATION_ASSUMPTION_VIOLATION, threshold=tau_ref
       d. Otherwise: => LOCAL_PERSONALIZE, threshold=tau_local

    Note: The pseudocode checks Gate B first (step 7), then Gate A (steps 8-11).
    But the state definitions in the table indicate that Gate B mismatch is
    a prerequisite for considering personalization.

    Args:
        reference: The federation reference threshold result
        gate_a: Gate A result for this client
        gate_b: Gate B result for this client

    Returns:
        FedCRGDecision with the selected state and threshold
    """
    tau_ref = reference.tau_ref

    # Step 7 in pseudocode: Check Gate B
    if gate_b.mismatch_state == GateBMismatchState.NO_MATERIAL_MISMATCH_DEMONSTRATED:
        return FedCRGDecision(
            state=FedCRGState.NO_MATERIAL_MISMATCH_DEMONSTRATED,
            selected_threshold=tau_ref,
            selected_source="reference",
            tie_count=0,
            reason_code=ReasonCode.GATE_B_NO_MISMATCH,
            gate_a_result=gate_a,
            gate_b_result=gate_b,
            reference_result=reference,
        )

    # Gate B has established mismatch (LOW or HIGH)
    # Step 3 in pseudocode (reordered for clarity): Check n_G minimum
    # Note: The pseudocode computes n_G_min from (a, gamma_B), which we do here
    n_g_min = compute_n_g_min(A(), GammaB()) if gate_b.a > 0 else 0
    
    if gate_b.n < n_g_min:
        return FedCRGDecision(
            state=FedCRGState.GATE_B_INSUFFICIENT,
            selected_threshold=tau_ref,
            selected_source="reference",
            tie_count=0,
            reason_code=ReasonCode.GATE_B_INSUFFICIENT_EVIDENCE,
            gate_a_result=gate_a,
            gate_b_result=gate_b,
            reference_result=reference,
        )

    # Step 8-11 in pseudocode: Gate A readiness check
    if not gate_a.ready:
        return FedCRGDecision(
            state=FedCRGState.CALIBRATION_DEFICIT,
            selected_threshold=tau_ref,
            selected_source="reference",
            tie_count=0,
            reason_code=ReasonCode.CALIBRATION_DEFICIT_STATE,
            gate_a_result=gate_a,
            gate_b_result=gate_b,
            reference_result=reference,
        )

    # Gate A is READY, check tie count
    if gate_a.tie_count > 1:
        return FedCRGDecision(
            state=FedCRGState.CALIBRATION_ASSUMPTION_VIOLATION,
            selected_threshold=tau_ref,
            selected_source="reference",
            tie_count=gate_a.tie_count,
            reason_code=ReasonCode.GATE_A_TIE_VIOLATION,
            gate_a_result=gate_a,
            gate_b_result=gate_b,
            reference_result=reference,
        )

    # All conditions met for local personalization
    return FedCRGDecision(
        state=FedCRGState.LOCAL_PERSONALIZE,
        selected_threshold=gate_a.tau_local,
        selected_source="local",
        tie_count=gate_a.tie_count,
        reason_code=ReasonCode.LOCAL_PERSONALIZATION_ADMITTED,
        gate_a_result=gate_a,
        gate_b_result=gate_b,
        reference_result=reference,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_state_from_conditions(
    gate_b_mismatch: GateBMismatchState,
    gate_a_ready: bool,
    n_g: int,
    gate_a_tie_count: int,
    a: float = A(),
    gamma_b: float = GammaB(),
) -> FedCRGState:
    """
    Determine FedCRG state from component conditions.

    This is a convenience function for quick state determination without
    constructing full result objects.

    Args:
        gate_b_mismatch: Gate B mismatch state
        gate_a_ready: Whether Gate A is ready
        n_g: Number of Gate B observations
        gate_a_tie_count: Tie count at selected Gate A threshold
        a: Lower band limit
        gamma_b: Gate B confidence level

    Returns:
        The FedCRG deployment state
    """
    n_g_min = compute_n_g_min(a, gamma_b) if a > 0 else 0

    if gate_b_mismatch == GateBMismatchState.NO_MATERIAL_MISMATCH_DEMONSTRATED:
        return FedCRGState.NO_MATERIAL_MISMATCH_DEMONSTRATED

    if n_g < n_g_min:
        return FedCRGState.GATE_B_INSUFFICIENT

    if not gate_a_ready:
        return FedCRGState.CALIBRATION_DEFICIT

    if gate_a_tie_count > 1:
        return FedCRGState.CALIBRATION_ASSUMPTION_VIOLATION

    return FedCRGState.LOCAL_PERSONALIZE


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "GateAMismatchState",
    "GateBMismatchState",
    "FedCRGState",
    "ReasonCode",
    # Types
    "FedCRGDecision",
    # Functions
    "decide_fedcrg",
    "get_state_from_conditions",
]
