"""
Oracle Baseline

Implements B10 (ORACLE-TEST) from Section 9.1.

Normative reference: Section 9.1, row B10
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from fedcrg.reference import Alpha, A, B
from fedcrg.baselines.quantile import QuantileBaseline, QuantileBaselineConfig


@dataclass(frozen=True, slots=True)
class OracleConfig:
    """
    Configuration for oracle baseline.
    
    Normative reference: Section 9.1, row B10
    """
    alpha: float = Alpha()
    a: float = A()  # 0.005
    b: float = B()  # 0.015


class OracleBaseline:
    """
    Baseline B10: ORACLE-TEST
    
    Unattainable diagnostic ceiling.
    
    For each client choose whichever of GLOBAL-Q99-FULL (B1), LOCAL-Q99-FULL (B2),
    or FedCRG gives smallest final-test band error.
    Break ties by higher TPR.
    
    This baseline uses final-test labels (B_k + A_test,k) and therefore
    cannot be used in actual deployment. It serves as an upper bound.
    
    Normative reference: Section 9.1, row B10
    """
    
    def __init__(
        self,
        config: OracleConfig = None,
        b1_thresholds: Dict[str, float] = None,
        b2_thresholds: Dict[str, float] = None,
        fedcrg_thresholds: Dict[str, float] = None,
    ):
        """
        Initialize oracle baseline.
        
        Args:
            config: Configuration
            b1_thresholds: Precomputed B1 thresholds
            b2_thresholds: Precomputed B2 thresholds
            fedcrg_thresholds: Precomputed FedCRG thresholds
        """
        if config is None:
            config = OracleConfig()
        self.config = config
        self.b1_thresholds = b1_thresholds or {}
        self.b2_thresholds = b2_thresholds or {}
        self.fedcrg_thresholds = fedcrg_thresholds or {}
    
    def compute_band_error(
        self,
        benign_scores: np.ndarray,
        attack_scores: np.ndarray,
        threshold: float,
    ) -> float:
        """
        Compute band error.
        
        Band error measures how far the actual FPR is from the target band [a, b].
        
        Args:
            benign_scores: Array of benign test scores
            attack_scores: Array of attack test scores
            threshold: Threshold for classification
            
        Returns:
            Band error (distance from band)
        """
        # Compute FPR
        fpr = float(np.mean(benign_scores > threshold)) if len(benign_scores) > 0 else 0.0
        
        # Compute band error
        if fpr < self.config.a:
            error = self.config.a - fpr
        elif fpr > self.config.b:
            error = fpr - self.config.b
        else:
            error = 0.0
        
        return error
    
    def compute_tpr(
        self,
        benign_scores: np.ndarray,
        attack_scores: np.ndarray,
        threshold: float,
    ) -> float:
        """
        Compute TPR (True Positive Rate / Recall).
        
        Args:
            benign_scores: Array of benign test scores
            attack_scores: Array of attack test scores
            threshold: Threshold for classification
            
        Returns:
            TPR
        """
        if len(attack_scores) == 0:
            return 0.0
        
        tp = int(np.sum(attack_scores > threshold))
        tpr = tp / len(attack_scores)
        
        return tpr
    
    def select_threshold(
        self,
        client_id: str,
        benign_test_scores: np.ndarray,
        attack_test_scores: np.ndarray,
    ) -> Tuple[float, str]:
        """
        Select threshold for a client that minimizes band error.
        
        Args:
            client_id: Client identifier
            benign_test_scores: Benign test scores (B_k)
            attack_test_scores: Attack test scores (A_test,k)
            
        Returns:
            Tuple of (selected_threshold, selected_baseline)
        """
        thresholds = {}
        
        if client_id in self.b1_thresholds:
            thresholds["B1"] = self.b1_thresholds[client_id]
        if client_id in self.b2_thresholds:
            thresholds["B2"] = self.b2_thresholds[client_id]
        if client_id in self.fedcrg_thresholds:
            thresholds["FedCRG"] = self.fedcrg_thresholds[client_id]
        
        if not thresholds:
            raise ValueError(f"No thresholds available for client {client_id}")
        
        # Evaluate each threshold
        results = []
        for baseline_id, threshold in thresholds.items():
            band_error = self.compute_band_error(
                benign_test_scores, attack_test_scores, threshold
            )
            tpr = self.compute_tpr(
                benign_test_scores, attack_test_scores, threshold
            )
            results.append((threshold, baseline_id, band_error, tpr))
        
        # Sort by band error (ascending), then by TPR (descending)
        results.sort(key=lambda x: (x[2], -x[3]))
        
        # Select first (smallest band error, highest TPR on tie)
        return results[0][0], results[0][1]
    
    def compute_thresholds(
        self,
        client_test_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Tuple[float, str]]:
        """
        Compute selected thresholds for all clients.
        
        Args:
            client_test_data: Dictionary mapping client_id to
                            (benign_test_scores, attack_test_scores)
            
        Returns:
            Dictionary mapping client_id to (threshold, baseline_id)
        """
        return {
            client_id: self.select_threshold(
                client_id,
                benign_test, attack_test
            )
            for client_id, (benign_test, attack_test) in client_test_data.items()
        }


# Singleton instance
B10_ORACLE_TEST = None


def verify_oracle() -> None:
    """Verify oracle baseline."""
    np.random.seed(42)
    
    # Create test data for 2 clients
    client_test_data = {
        "nb01": (np.random.randn(1000), np.random.randn(1000) + 2.0),
        "nb02": (np.random.randn(1000), np.random.randn(1000) + 2.0),
    }
    
    # Create thresholds
    b1_thresholds = {"nb01": 0.5, "nb02": 0.5}
    b2_thresholds = {"nb01": 1.0, "nb02": 1.0}
    fedcrg_thresholds = {"nb01": 1.5, "nb02": 1.5}
    
    oracle = OracleBaseline(
        OracleConfig(),
        b1_thresholds, b2_thresholds, fedcrg_thresholds
    )
    
    oracle_thresholds = oracle.compute_thresholds(client_test_data)
    
    assert len(oracle_thresholds) == 2
    
    # Verify that selected thresholds are from the provided options
    for client_id, (threshold, baseline_id) in oracle_thresholds.items():
        assert baseline_id in ["B1", "B2", "FedCRG"]
        assert threshold in [0.5, 1.0, 1.5]
    
    print("Oracle baseline verification passed.")


if __name__ == "__main__":
    verify_oracle()
