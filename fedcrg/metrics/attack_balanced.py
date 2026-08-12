"""
FedCRG Attack-Balanced TPR Metrics Module

Implements Attack-Balanced Macro-TPR per Section 10.1 of the FedCRG Roadmap v2.0.

Key definitions:
- ABTPR_kj = TP_kj / (TP_kj + FN_kj) for each attack group j
- ABTPR_k = (1/|A_k|) * sum_{j in A_k} ABTPR_kj
- ABMacroTPR = (1/K) * sum_{k=1}^K ABTPR_k

This gives each attack group equal weight within a client and each client
equal weight in the federation.

Critical requirement (H3):
Any claimed FedCRG reliability gain must incur no more than a 3.0 percentage-point
absolute loss in Attack-Balanced Macro-TPR relative to the locked benign-only
utility anchor max(GLOBAL-Q99-FULL, LOCAL-Q99-FULL, SHRINKAGE).

Normative reference: Section 10.1, Section 1357-1376
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    pass


# =============================================================================
# TYPE ALIASES
# =============================================================================

ABTPR = float
ABMacroTPR = float


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass(frozen=True, slots=True)
class AttackGroupTPR:
    """
    TPR for a single attack group.

    ABTPR_kj = TP_kj / (TP_kj + FN_kj)

    Attributes:
        client_id: Client identifier
        attack_group: Attack group identifier (e.g., "gafgyt_combo", "mirai_syn")
        tp: True positives
        fn: False negatives
        tpr: True positive rate (TP / (TP + FN))
    """

    client_id: str
    attack_group: str
    tp: int
    fn: int
    tpr: float

    @classmethod
    def from_counts(
        cls,
        client_id: str,
        attack_group: str,
        tp: int,
        fn: int,
    ) -> "AttackGroupTPR":
        """
        Create AttackGroupTPR from TP and FN counts.

        Args:
            client_id: Client identifier
            attack_group: Attack group identifier
            tp: True positives
            fn: False negatives

        Returns:
            AttackGroupTPR
        """
        total_positives = tp + fn
        if total_positives == 0:
            tpr = float('nan')
        else:
            tpr = tp / total_positives

        return cls(
            client_id=client_id,
            attack_group=attack_group,
            tp=tp,
            fn=fn,
            tpr=tpr,
        )


@dataclass(frozen=True, slots=True)
class ClientABTPR:
    """
    Attack-Balanced TPR for a single client.

    ABTPR_k = (1/|A_k|) * sum_{j in A_k} ABTPR_kj

    This gives each attack group equal weight within a client.

    Attributes:
        client_id: Client identifier
        attack_groups: List of attack groups present for this client
        group_tprs: Dictionary mapping attack group to AttackGroupTPR
        abmacro_tpr: Attack-balanced macro TPR for this client
    """

    client_id: str
    attack_groups: List[str]
    group_tprs: Dict[str, AttackGroupTPR]
    abmacro_tpr: float

    @property
    def n_groups(self) -> int:
        """Number of attack groups."""
        return len(self.attack_groups)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "client_id": self.client_id,
            "attack_groups": self.attack_groups,
            "group_tprs": {
                ag: {"tp": gtp.tp, "fn": gtp.fn, "tpr": gtp.tpr}
                for ag, gtp in self.group_tprs.items()
            },
            "abmacro_tpr": self.abmacro_tpr,
        }


@dataclass(frozen=True, slots=True)
class AttackBalancedTPR:
    """
    Attack-Balanced Macro-TPR for all clients.

    ABMacroTPR = (1/K) * sum_{k=1}^K ABTPR_k

    This gives each client equal weight in the federation.
    Missing attack types are absent, not zero.

    Attributes:
        abmacro_tpr: Attack-balanced macro TPR value
        n_clients: Number of clients
        client_abmacro_tprs: Dictionary mapping client_id to ClientABTPR
        utility_anchor: The benign-only utility anchor (max of GLOBAL-Q99-FULL, LOCAL-Q99-FULL, SHRINKAGE)
        margin: Non-inferiority margin (default: 0.03)
    """

    abmacro_tpr: float
    n_clients: int
    client_abmacro_tprs: Dict[str, ClientABTPR]
    utility_anchor: Optional[float] = None
    margin: float = 0.03  # 3 percentage points

    @property
    def is_utility_preserving(self) -> bool:
        """
        Check if the result meets the utility-preserving criterion.

        A claimed operating-reliability gain is utility-preserving only when:
        ABMacroTPR_FedCRG - U_anchor >= -0.03
        
        Or equivalently: ABMacroTPR_FedCRG >= U_anchor - 0.03
        """
        if self.utility_anchor is None:
            return False
        return self.abmacro_tpr >= self.utility_anchor - self.margin

    @property
    def utility_diff(self) -> Optional[float]:
        """
        Difference from utility anchor.

        Returns:
            ABMacroTPR - utility_anchor, or None if anchor not set
        """
        if self.utility_anchor is None:
            return None
        return self.abmacro_tpr - self.utility_anchor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "abmacro_tpr": self.abmacro_tpr,
            "n_clients": self.n_clients,
            "client_abmacro_tprs": {
                cid: client_abtpr.to_dict()
                for cid, client_abtpr in self.client_abmacro_tprs.items()
            },
            "utility_anchor": self.utility_anchor,
            "margin": self.margin,
            "is_utility_preserving": self.is_utility_preserving,
            "utility_diff": self.utility_diff,
        }


# =============================================================================
# COMPUTATION FUNCTIONS
# =============================================================================


def compute_attack_group_tpr(
    client_id: str,
    attack_group: str,
    tp: int,
    fn: int,
) -> AttackGroupTPR:
    """
    Compute TPR for a single attack group.

    Args:
        client_id: Client identifier
        attack_group: Attack group identifier
        tp: True positives
        fn: False negatives

    Returns:
        AttackGroupTPR
    """
    return AttackGroupTPR.from_counts(client_id, attack_group, tp, fn)


def compute_client_abmacro_tpr(
    client_id: str,
    attack_group_tprs: List[AttackGroupTPR],
) -> ClientABTPR:
    """
    Compute attack-balanced TPR for a single client.

    ABTPR_k = (1/|A_k|) * sum_{j in A_k} ABTPR_kj

    Args:
        client_id: Client identifier
        attack_group_tprs: List of AttackGroupTPR for this client

    Returns:
        ClientABTPR
    """
    if not attack_group_tprs:
        raise ValueError(f"No attack groups for client {client_id}")

    attack_groups = [agt.attack_group for agt in attack_group_tprs]
    group_tprs = {agt.attack_group: agt for agt in attack_group_tprs}

    # Compute mean TPR across attack groups
    valid_tprs = [agt.tpr for agt in attack_group_tprs if not np.isnan(agt.tpr)]

    if not valid_tprs:
        abmacro_tpr = float('nan')
    else:
        abmacro_tpr = np.mean(valid_tprs)

    return ClientABTPR(
        client_id=client_id,
        attack_groups=attack_groups,
        group_tprs=group_tprs,
        abmacro_tpr=abmacro_tpr,
    )


def compute_abmacro_tpr(
    client_abmacro_tprs: List[ClientABTPR],
    utility_anchor: Optional[float] = None,
    margin: float = 0.03,
) -> AttackBalancedTPR:
    """
    Compute Attack-Balanced Macro-TPR for all clients.

    ABMacroTPR = (1/K) * sum_{k=1}^K ABTPR_k

    Args:
        client_abmacro_tprs: List of ClientABTPR for all clients
        utility_anchor: Optional benign-only utility anchor
        margin: Non-inferiority margin (default: 0.03)

    Returns:
        AttackBalancedTPR
    """
    n_clients = len(client_abmacro_tprs)
    if n_clients == 0:
        raise ValueError("No clients provided")

    client_abtpr_dict = {cat.client_id: cat for cat in client_abmacro_tprs}

    # Compute mean ABTPR across clients
    valid_abmacro_tprs = [
        cat.abmacro_tpr
        for cat in client_abmacro_tprs
        if not np.isnan(cat.abmacro_tpr)
    ]

    if not valid_abmacro_tprs:
        abmacro_tpr = float('nan')
    else:
        abmacro_tpr = np.mean(valid_abmacro_tprs)

    return AttackBalancedTPR(
        abmacro_tpr=abmacro_tpr,
        n_clients=n_clients,
        client_abmacro_tprs=client_abtpr_dict,
        utility_anchor=utility_anchor,
        margin=margin,
    )


# =============================================================================
# BATCH COMPUTATION FROM SCRES
# =============================================================================


def compute_client_abmacro_tpr_from_scores(
    client_id: str,
    attack_groups: List[str],
    attack_group_scores: Dict[str, npt.NDArray[np.float64]],
    benign_scores: npt.NDArray[np.float64],
    threshold: float,
) -> ClientABTPR:
    """
    Compute attack-balanced TPR for a client from scores.

    Args:
        client_id: Client identifier
        attack_groups: List of attack group identifiers
        attack_group_scores: Dictionary mapping attack group to scores
        benign_scores: Benign scores for this client
        threshold: Decision threshold (score > threshold = anomaly)

    Returns:
        ClientABTPR
    """
    attack_group_tprs = []

    for ag in attack_groups:
        if ag not in attack_group_scores:
            # Missing attack type - skip (absent, not zero)
            continue

        ag_scores = attack_group_scores[ag]

        # Count TP and FN for this attack group
        tp = int(np.sum(ag_scores > threshold))
        fn = int(np.sum(ag_scores <= threshold))

        agt = compute_attack_group_tpr(client_id, ag, tp, fn)
        attack_group_tprs.append(agt)

    return compute_client_abmacro_tpr(client_id, attack_group_tprs)


def compute_abmacro_tpr_from_scores(
    clients_data: Dict[str, Tuple[List[str], Dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]],
    threshold: float,
) -> AttackBalancedTPR:
    """
    Compute Attack-Balanced Macro-TPR from scores for all clients.

    Args:
        clients_data: Dictionary mapping client_id to (attack_groups, attack_group_scores, benign_scores)
        threshold: Decision threshold

    Returns:
        AttackBalancedTPR
    """
    client_abmacro_tprs = []

    for client_id, (attack_groups, attack_group_scores, benign_scores) in clients_data.items():
        cat = compute_client_abmacro_tpr_from_scores(
            client_id, attack_groups, attack_group_scores, benign_scores, threshold
        )
        client_abmacro_tprs.append(cat)

    return compute_abmacro_tpr(client_abmacro_tprs)


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================


def verify_attack_balanced_metrics() -> bool:
    """
    Verify attack-balanced TPR metrics implementation.

    Tests:
    - Attack group TPR computation
    - Client ABMacroTPR computation
    - Federated ABMacroTPR computation
    - Utility anchor comparison
    - Edge cases (missing attack types, no positives)
    """
    import numpy as np

    # Test data: 2 clients, each with 2 attack groups
    # Client 01: groups A and B
    # Client 02: groups A, B, and C (C is missing for client 01)

    # Client 01, Group A: TP=80, FN=20 => TPR=0.8
    agt_01_a = compute_attack_group_tpr("client_01", "A", 80, 20)
    assert abs(agt_01_a.tpr - 0.8) < 1e-10

    # Client 01, Group B: TP=90, FN=10 => TPR=0.9
    agt_01_b = compute_attack_group_tpr("client_01", "B", 90, 10)
    assert abs(agt_01_b.tpr - 0.9) < 1e-10

    # Client 01 ABMacroTPR = (0.8 + 0.9) / 2 = 0.85
    cat_01 = compute_client_abmacro_tpr("client_01", [agt_01_a, agt_01_b])
    assert abs(cat_01.abmacro_tpr - 0.85) < 1e-10

    # Client 02, Group A: TP=70, FN=30 => TPR=0.7
    agt_02_a = compute_attack_group_tpr("client_02", "A", 70, 30)
    assert abs(agt_02_a.tpr - (70/100)) < 1e-10

    # Client 02, Group B: TP=85, FN=15 => TPR=0.85
    agt_02_b = compute_attack_group_tpr("client_02", "B", 85, 15)
    assert abs(agt_02_b.tpr - 0.85) < 1e-10

    # Client 02, Group C: TP=95, FN=5 => TPR=0.95
    agt_02_c = compute_attack_group_tpr("client_02", "C", 95, 5)
    assert abs(agt_02_c.tpr - 0.95) < 1e-10

    # Client 02 ABMacroTPR = (0.7 + 0.85 + 0.95) / 3
    expected_02 = (0.7 + 0.85 + 0.95) / 3
    cat_02 = compute_client_abmacro_tpr("client_02", [agt_02_a, agt_02_b, agt_02_c])
    assert abs(cat_02.abmacro_tpr - expected_02) < 1e-10

    # Federated ABMacroTPR = (0.85 + 0.8333...) / 2
    expected_fed = (0.85 + expected_02) / 2
    abmacro_result = compute_abmacro_tpr([cat_01, cat_02])
    assert abs(abmacro_result.abmacro_tpr - expected_fed) < 1e-10

    # Test utility anchor
    utility_anchor = 0.82
    abmacro_with_anchor = compute_abmacro_tpr(
        [cat_01, cat_02],
        utility_anchor=utility_anchor,
        margin=0.03,
    )
    assert abmacro_with_anchor.utility_anchor == utility_anchor
    assert abmacro_with_anchor.margin == 0.03

    # Check if utility-preserving
    # ABMacroTPR_FedCRG >= U_anchor - 0.03
    is_preserving = (abmacro_with_anchor.abmacro_tpr >= utility_anchor - 0.03)
    assert abmacro_with_anchor.is_utility_preserving == is_preserving

    # Test edge case: no attack groups
    try:
        compute_client_abmacro_tpr("client_03", [])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Test edge case: all FN (TPR = 0)
    agt_zero = compute_attack_group_tpr("client_04", "A", 0, 100)
    assert agt_zero.tpr == 0.0

    # Test edge case: all TP (TPR = 1)
    agt_one = compute_attack_group_tpr("client_05", "A", 100, 0)
    assert agt_one.tpr == 1.0

    # Test edge case: no positives (TPR = NaN)
    agt_nan = compute_attack_group_tpr("client_06", "A", 0, 0)
    assert np.isnan(agt_nan.tpr)

    print("Attack-balanced TPR metrics verification passed.")
    return True


if __name__ == "__main__":
    verify_attack_balanced_metrics()
