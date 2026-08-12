"""
FedCRG Core Algorithm

Implements the complete FedCRG decision algorithm per Section 5.

Normative reference: Section 5
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fedcrg.gate_a import compute_gate_a, GateAResult
from fedcrg.gate_b import compute_gate_b, GateBResult
from fedcrg.reference import build_reference_threshold, Alpha, A, B, GammaB, PrimaryNGMin
from fedcrg.states import (
    FedCRGState,
    GateBMismatchState,
)


@dataclass(frozen=True, slots=True)
class FedCRGConfig:
    """
    Configuration for FedCRG execution.
    
    Normative reference: Section 5
    """
    alpha: float = Alpha()
    rho: float = 0.50
    gamma_a: float = 0.95
    gamma_b: float = GammaB()
    
    # Gate B minimum
    n_g_min: int = PrimaryNGMin  # 736
    
    # Multiplicity check
    check_multiplicity: bool = True
    
    # Tie handling for Gate A
    # ties resolved in favor of larger r (more conservative threshold)
    tie_behavior: str = "larger_r"


@dataclass(frozen=True, slots=True)
class ClientGateResult:
    """
    Result of FedCRG gate computation for a single client.
    
    Normative reference: Section 5
    """
    client_id: str
    
    # Gate B results
    gate_b_state: str  # "LOW_MISMATCH", "HIGH_MISMATCH", "NO_MATERIAL_MISMATCH_DEMONSTRATED", or "GATE_B_INSUFFICIENT"
    gate_b_x: int  # count of g > tau_ref in G_k
    gate_b_n: int  # size of G_k
    gate_b_lower: float  # cp_lower from Clopper-Pearson
    gate_b_upper: float  # cp_upper from Clopper-Pearson
    
    # Gate A results
    gate_a_state: str  # "READY" or "NOT_READY"
    gate_a_max_p: float  # coverage_probability
    gate_a_r_star: int  # rank (r*)
    gate_a_tau_local: float  # local threshold
    gate_a_multiplicity: int  # tie_count at tau_local
    
    # FedCRG decision
    admission_state: str  # Final admission state
    threshold: float  # Selected threshold
    
    # Hash for verification
    hash: str = field(default="")
    
    def __post_init__(self):
        """Compute hash."""
        object.__setattr__(
            self, "hash",
            self._compute_hash()
        )
    
    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of the result."""
        data = {
            "client_id": self.client_id,
            "gate_b_state": self.gate_b_state,
            "gate_b_x": self.gate_b_x,
            "gate_b_n": self.gate_b_n,
            "gate_b_lower": self.gate_b_lower,
            "gate_b_upper": self.gate_b_upper,
            "gate_a_state": self.gate_a_state,
            "gate_a_max_p": self.gate_a_max_p,
            "gate_a_r_star": self.gate_a_r_star,
            "gate_a_tau_local": self.gate_a_tau_local,
            "gate_a_multiplicity": self.gate_a_multiplicity,
            "admission_state": self.admission_state,
            "threshold": self.threshold,
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "client_id": self.client_id,
            "gate_b_state": self.gate_b_state,
            "gate_b_x": self.gate_b_x,
            "gate_b_n": self.gate_b_n,
            "gate_b_lower": self.gate_b_lower,
            "gate_b_upper": self.gate_b_upper,
            "gate_a_state": self.gate_a_state,
            "gate_a_max_p": self.gate_a_max_p,
            "gate_a_r_star": self.gate_a_r_star,
            "gate_a_tau_local": self.gate_a_tau_local,
            "gate_a_multiplicity": self.gate_a_multiplicity,
            "admission_state": self.admission_state,
            "threshold": self.threshold,
            "hash": self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientGateResult":
        """Create from dictionary."""
        return cls(
            client_id=data["client_id"],
            gate_b_state=data["gate_b_state"],
            gate_b_x=data["gate_b_x"],
            gate_b_n=data["gate_b_n"],
            gate_b_lower=data["gate_b_lower"],
            gate_b_upper=data["gate_b_upper"],
            gate_a_state=data["gate_a_state"],
            gate_a_max_p=data["gate_a_max_p"],
            gate_a_r_star=data["gate_a_r_star"],
            gate_a_tau_local=data["gate_a_tau_local"],
            gate_a_multiplicity=data["gate_a_multiplicity"],
            admission_state=data["admission_state"],
            threshold=data["threshold"],
        )


@dataclass(frozen=True, slots=True)
class FedCRGResult:
    """
    Complete result of FedCRG execution for all clients.
    
    Normative reference: Section 5
    """
    tau_ref: float  # Federation reference threshold
    n_r: int  # Total reference scores
    q_ref: int  # Reference rank
    
    client_results: Dict[str, ClientGateResult]  # client_id -> ClientGateResult
    
    # Summary statistics
    n_clients_admitted: int
    n_clients_calibration_deficit: int
    n_clients_no_mismatch: int
    n_clients_assumption_violation: int
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        states = {}
        for cid, result in self.client_results.items():
            state = result.admission_state
            states[state] = states.get(state, 0) + 1
        
        return {
            "total_clients": len(self.client_results),
            "tau_ref": self.tau_ref,
            "n_r": self.n_r,
            "q_ref": self.q_ref,
            "state_counts": states,
            "timestamp": self.timestamp,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tau_ref": self.tau_ref,
            "n_r": self.n_r,
            "q_ref": self.q_ref,
            "client_results": {
                cid: cr.to_dict() for cid, cr in self.client_results.items()
            },
            "n_clients_admitted": self.n_clients_admitted,
            "n_clients_calibration_deficit": self.n_clients_calibration_deficit,
            "n_clients_no_mismatch": self.n_clients_no_mismatch,
            "n_clients_assumption_violation": self.n_clients_assumption_violation,
            "timestamp": self.timestamp,
        }
    
    def to_json(self, path: Path) -> None:
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FedCRGResult":
        """Create from dictionary."""
        client_results = {
            cid: ClientGateResult.from_dict(cr_data)
            for cid, cr_data in data["client_results"].items()
        }
        return cls(
            tau_ref=data["tau_ref"],
            n_r=data["n_r"],
            q_ref=data["q_ref"],
            client_results=client_results,
            n_clients_admitted=data["n_clients_admitted"],
            n_clients_calibration_deficit=data["n_clients_calibration_deficit"],
            n_clients_no_mismatch=data["n_clients_no_mismatch"],
            n_clients_assumption_violation=data["n_clients_assumption_violation"],
            timestamp=data.get("timestamp", ""),
        )


class FedCRG:
    """
    FedCRG - Federated Calibration Readiness Gate.
    
    Implements the complete FedCRG algorithm per Section 5.
    
    Normative reference: Section 5
    """
    
    def __init__(self, config: FedCRGConfig = None):
        """
        Initialize FedCRG.
        
        Args:
            config: FedCRG configuration
        """
        if config is None:
            config = FedCRGConfig()
        self.config = config
    
    def compute_reference_threshold(
        self,
        reference_scores_by_client: Dict[str, np.ndarray],
    ) -> Tuple[float, int, int]:
        """
        Compute federation reference threshold.
        
        Args:
            reference_scores_by_client: Dictionary mapping client_id to R_k scores
            
        Returns:
            Tuple of (tau_ref, n_r, q_ref)
            
        Normative reference: Section 5.1
        """
        result = build_reference_threshold(
            reference_scores_by_client,
            alpha=self.config.alpha,
        )
        return result.tau_ref, result.n_r, result.q_ref
    
    def compute_gate_b(
        self,
        gate_scores: np.ndarray,
        tau_ref: float,
    ) -> GateBResult:
        """
        Compute Gate B for a client.
        
        Args:
            gate_scores: G_k scores
            tau_ref: Federation reference threshold
            
        Returns:
            GateBResult
            
        Normative reference: Section 5.3
        """
        return compute_gate_b(
            gate_scores,
            tau_ref,
            a=A(),
            b=B(),
            gamma_b=self.config.gamma_b,
        )
    
    def compute_gate_a(
        self,
        calibration_scores: np.ndarray,
    ) -> GateAResult:
        """
        Compute Gate A for a client.
        
        Args:
            calibration_scores: C_k scores
            
        Returns:
            GateAResult
            
        Normative reference: Section 5.2
        """
        return compute_gate_a(
            calibration_scores,
            alpha=self.config.alpha,
            rho=self.config.rho,
            gamma_a=self.config.gamma_a,
        )
    
    def compute_client(
        self,
        client_id: str,
        reference_scores: np.ndarray,
        gate_scores: np.ndarray,
        calibration_scores: np.ndarray,
        tau_ref: float,
    ) -> ClientGateResult:
        """
        Compute FedCRG result for a single client.
        
        Implements the algorithm from Section 5.4 pseudocode.
        
        Args:
            client_id: Client identifier
            reference_scores: R_k scores (for reference threshold verification)
            gate_scores: G_k scores
            calibration_scores: C_k scores
            tau_ref: Federation reference threshold
            
        Returns:
            ClientGateResult with all gate computations and decision
            
        Normative reference: Section 5.4
        """
        # Step 3: Check Gate B minimum
        if len(gate_scores) < self.config.n_g_min:
            # Create a minimal GateBResult for the INSUFFICIENT case
            from fedcrg.states import GateBMismatchState
            gate_b_result = GateBResult(
                n=len(gate_scores),
                x=0,
                fpr_hat=0.0,
                cp_lower=0.0,
                cp_upper=0.0,
                p_low=1.0,
                p_high=1.0,
                mismatch_state=GateBMismatchState.NO_MATERIAL_MISMATCH_DEMONSTRATED,
                a=A(),
                b=B(),
                gamma_b=self.config.gamma_b,
                n_g_min=self.config.n_g_min,
                gate_scores=None,
            )
        else:
            gate_b_result = self.compute_gate_b(gate_scores, tau_ref)
        
        # Step 8: Compute Gate A
        gate_a_result = self.compute_gate_a(calibration_scores)
        
        # Extract values
        gate_b_mismatch = gate_b_result.mismatch_state
        gate_b_state = gate_b_mismatch.name
        gate_b_x = gate_b_result.x
        gate_b_n = gate_b_result.n
        gate_b_lower = gate_b_result.cp_lower
        gate_b_upper = gate_b_result.cp_upper
        
        gate_a_state = "READY" if gate_a_result.ready else "NOT_READY"
        gate_a_max_p = gate_a_result.coverage_probability
        gate_a_r_star = gate_a_result.rank
        gate_a_tau_local = gate_a_result.tau_local if gate_a_result.tau_local is not None else 0.0
        gate_a_multiplicity = gate_a_result.tie_count
        
        # Step 11: Check multiplicity
        multiplicity_violation = False
        if self.config.check_multiplicity and gate_a_multiplicity > 1:
            multiplicity_violation = True
        
        # Determine admission state and threshold
        # Per Section 5.4 pseudocode:
        # 7. mismatch = LOW if U<a; HIGH if L>b; otherwise NONE
        # 11. If mismatch == NONE: NO_MATERIAL_MISMATCH_DEMONSTRATED
        # 9. If max(P_r) < gamma_A: CALIBRATION_DEFICIT
        # 12. If tie_count > 1: CALIBRATION_ASSUMPTION_VIOLATION
        # 13. return LOCAL_PERSONALIZE, threshold=tau_local
        
        mismatch = gate_b_mismatch.name if gate_b_mismatch else "NONE"
        
        # Decision logic
        if mismatch == "NONE":
            admission_state = "NO_MATERIAL_MISMATCH_DEMONSTRATED"
            threshold = tau_ref
        elif gate_a_state != "READY" or gate_a_max_p < self.config.gamma_a:
            admission_state = "CALIBRATION_DEFICIT"
            threshold = tau_ref
        elif multiplicity_violation:
            admission_state = "CALIBRATION_ASSUMPTION_VIOLATION"
            threshold = tau_ref
        else:
            admission_state = "LOCAL_PERSONALIZE"
            threshold = gate_a_tau_local
        
        return ClientGateResult(
            client_id=client_id,
            gate_b_state=gate_b_state,
            gate_b_x=gate_b_x,
            gate_b_n=gate_b_n,
            gate_b_lower=gate_b_lower,
            gate_b_upper=gate_b_upper,
            gate_a_state=gate_a_state,
            gate_a_max_p=gate_a_max_p,
            gate_a_r_star=gate_a_r_star,
            gate_a_tau_local=gate_a_tau_local,
            gate_a_multiplicity=gate_a_multiplicity,
            admission_state=admission_state,
            threshold=threshold,
        )
    
    def compute(
        self,
        reference_scores_by_client: Dict[str, np.ndarray],
        gate_scores_by_client: Dict[str, np.ndarray],
        calibration_scores_by_client: Dict[str, np.ndarray],
    ) -> FedCRGResult:
        """
        Compute FedCRG for all clients.
        
        Args:
            reference_scores_by_client: Dictionary mapping client_id to R_k scores
            gate_scores_by_client: Dictionary mapping client_id to G_k scores
            calibration_scores_by_client: Dictionary mapping client_id to C_k scores
            
        Returns:
            FedCRGResult with complete results
            
        Normative reference: Section 5
        """
        # Step 1: Compute reference threshold
        tau_ref, n_r, q_ref = self.compute_reference_threshold(
            reference_scores_by_client
        )
        
        # Validate client sets match
        client_ids = set(reference_scores_by_client.keys())
        if client_ids != set(gate_scores_by_client.keys()):
            raise ValueError("Reference and gate client sets don't match")
        if client_ids != set(calibration_scores_by_client.keys()):
            raise ValueError("Reference and calibration client sets don't match")
        
        # Compute for each client
        client_results = {}
        for client_id in client_ids:
            result = self.compute_client(
                client_id=client_id,
                reference_scores=reference_scores_by_client[client_id],
                gate_scores=gate_scores_by_client[client_id],
                calibration_scores=calibration_scores_by_client[client_id],
                tau_ref=tau_ref,
            )
            client_results[client_id] = result
        
        # Count summary statistics
        n_admitted = sum(
            1 for r in client_results.values()
            if r.admission_state == "LOCAL_PERSONALIZE"
        )
        n_deficit = sum(
            1 for r in client_results.values()
            if r.admission_state == "CALIBRATION_DEFICIT"
        )
        n_no_mismatch = sum(
            1 for r in client_results.values()
            if r.admission_state == "NO_MATERIAL_MISMATCH_DEMONSTRATED"
        )
        n_assumption_violation = sum(
            1 for r in client_results.values()
            if r.admission_state == "CALIBRATION_ASSUMPTION_VIOLATION"
        )
        
        return FedCRGResult(
            tau_ref=tau_ref,
            n_r=n_r,
            q_ref=q_ref,
            client_results=client_results,
            n_clients_admitted=n_admitted,
            n_clients_calibration_deficit=n_deficit,
            n_clients_no_mismatch=n_no_mismatch,
            n_clients_assumption_violation=n_assumption_violation,
        )


def verify_fedcrg() -> None:
    """Verify FedCRG implementation."""
    import numpy as np
    from fedcrg.gate_a import compute_gate_a
    from fedcrg.gate_b import compute_gate_b
    
    fedcrg = FedCRG()
    
    # Create test data for 2 clients
    np.random.seed(42)
    
    reference_scores = {
        "nb01": np.random.randn(500),
        "nb02": np.random.randn(500),
    }
    gate_scores = {
        "nb01": np.random.randn(3000),
        "nb02": np.random.randn(3000),
    }
    calibration_scores = {
        "nb01": np.random.randn(2000),
        "nb02": np.random.randn(2000),
    }
    
    # Compute
    result = fedcrg.compute(reference_scores, gate_scores, calibration_scores)
    
    assert result.tau_ref is not None
    assert result.n_r == 1000  # 2 * 500
    assert result.q_ref == 991  # min(1000, ceil(1001 * 0.99)) = ceil(990.99) = 991
    assert len(result.client_results) == 2
    
    # Check each client result
    for cid, cr in result.client_results.items():
        assert cr.client_id == cid
        assert cr.gate_b_n == 3000
        assert cr.gate_a_multiplicity >= 0
        assert cr.admission_state in [
            "LOCAL_PERSONALIZE",
            "CALIBRATION_DEFICIT",
            "NO_MATERIAL_MISMATCH_DEMONSTRATED",
            "CALIBRATION_ASSUMPTION_VIOLATION",
        ]
    
    # Test reference threshold computation
    tau_ref, n_r, q_ref = fedcrg.compute_reference_threshold(reference_scores)
    assert n_r == 1000
    assert q_ref == 991
    
    print("FedCRG verification passed.")


if __name__ == "__main__":
    verify_fedcrg()
