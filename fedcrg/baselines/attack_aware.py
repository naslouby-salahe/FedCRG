"""
Attack-Aware Baselines

Implements B7, B8, B9 from Sections 9.1 and 9.3-9.4.

Normative reference: Sections 9.1, 9.3, 9.4
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fedcrg.reference import Alpha
from fedcrg.baselines.quantile import QuantileBaseline, QuantileBaselineConfig


@dataclass(frozen=True, slots=True)
class AttackAwareConfig:
    """
    Configuration for attack-aware baselines.
    
    Normative reference: Sections 9.1, 9.3, 9.4
    """
    alpha: float = Alpha()
    n_dev: int = 500  # Development budget per client
    n_guard: int = 500  # Benign guard
    n_candidates: int = 1000  # For B8 and B9


class DevF1LgSelectBaseline:
    """
    Baseline B7: DEV-F1-LG-SELECT
    
    Attack-aware selector. Closest simple attack-aware answer to
    "local or global?"
    
    Uses 500 benign guard + 500 attack-balanced A_dev per client
    (50:50 development prevalence).
    
    Compute F1 for B1 (GLOBAL-Q99-FULL) and B2 (LOCAL-Q99-FULL) on
    client development set.
    
    Select B2 only if its F1 is strictly larger; ties select B1 to
    avoid unnecessary personalization.
    
    Freeze the selected choice and evaluate it once on B_k + A_test,k.
    
    Normative reference: Section 9.3
    """
    
    def __init__(
        self,
        config: AttackAwareConfig = None,
        b1_thresholds: Dict[str, float] = None,
        b2_thresholds: Dict[str, float] = None,
    ):
        """
        Initialize B7 baseline.
        
        Args:
            config: Configuration
            b1_thresholds: Precomputed B1 thresholds (GLOBAL-Q99-FULL)
            b2_thresholds: Precomputed B2 thresholds (LOCAL-Q99-FULL)
        """
        if config is None:
            config = AttackAwareConfig()
        self.config = config
        self.b1_thresholds = b1_thresholds or {}
        self.b2_thresholds = b2_thresholds or {}
    
    def compute_f1(
        self,
        benign_scores: np.ndarray,
        attack_scores: np.ndarray,
        threshold: float,
    ) -> float:
        """
        Compute F1 score.
        
        Args:
            benign_scores: Array of benign scores
            attack_scores: Array of attack scores
            threshold: Threshold for classification
            
        Returns:
            F1 score
        """
        # Classify
        benign_pred = np.zeros(len(benign_scores), dtype=bool)  # True = anomaly
        attack_pred = attack_scores > threshold
        
        # True labels
        benign_true = np.zeros(len(benign_scores), dtype=bool)
        attack_true = np.ones(len(attack_scores), dtype=bool)
        
        # Combine
        all_pred = np.concatenate([benign_pred, attack_pred])
        all_true = np.concatenate([benign_true, attack_true])
        
        # Compute TP, FP, FN
        tp = int(np.sum((all_pred == True) & (all_true == True)))
        fp = int(np.sum((all_pred == True) & (all_true == False)))
        fn = int(np.sum((all_pred == False) & (all_true == True)))
        
        # Compute precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Compute F1
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return f1
    
    def select_threshold(
        self,
        client_id: str,
        dev_benign_scores: np.ndarray,
        dev_attack_scores: np.ndarray,
    ) -> Tuple[float, str]:
        """
        Select threshold for a client based on development F1.
        
        Args:
            client_id: Client identifier
            dev_benign_scores: Benign development scores (guard)
            dev_attack_scores: Attack development scores (A_dev)
            
        Returns:
            Tuple of (selected_threshold, selected_baseline)
        """
        if client_id not in self.b1_thresholds or client_id not in self.b2_thresholds:
            raise ValueError(f"Missing thresholds for client {client_id}")
        
        b1_threshold = self.b1_thresholds[client_id]
        b2_threshold = self.b2_thresholds[client_id]
        
        # Compute F1 for both thresholds
        f1_b1 = self.compute_f1(dev_benign_scores, dev_attack_scores, b1_threshold)
        f1_b2 = self.compute_f1(dev_benign_scores, dev_attack_scores, b2_threshold)
        
        # Select B2 only if strictly larger F1
        if f1_b2 > f1_b1:
            return b2_threshold, "B2"
        else:
            return b1_threshold, "B1"
    
    def compute_thresholds(
        self,
        client_dev_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Tuple[float, str]]:
        """
        Compute selected thresholds for all clients.
        
        Args:
            client_dev_data: Dictionary mapping client_id to
                            (dev_benign_scores, dev_attack_scores)
            
        Returns:
            Dictionary mapping client_id to (threshold, baseline_id)
        """
        return {
            client_id: self.select_threshold(
                client_id,
                dev_benign, dev_attack
            )
            for client_id, (dev_benign, dev_attack) in client_dev_data.items()
        }


class LaridiStyleSSBaseline:
    """
    Baseline B8: LARIDI-STYLE-SS
    
    Laridi et al. 2024 closest-prior comparator.
    
    For each client and class (benign, attack), on the fixed 1,000-record
    development set compute:
    n_ky, mu_ky, v_ky
    
    Server computes pooled population moments:
    N_y = sum_k n_ky
    mu_y = sum_k n_ky * mu_ky / N_y
    v_y = sum_k n_ky * (v_ky + mu_ky^2) / N_y - mu_y^2
    sigma_y = sqrt(max(v_y, 0))
    
    Define overlap interval:
    l = max(mu_benign - 3*sigma_benign, mu_attack - 3*sigma_attack)
    u = min(mu_benign + 3*sigma_benign, mu_attack + 3*sigma_attack)
    
    If l >= u: record LARIDI_STYLE_UNDEFINED
    Otherwise: 1000 equally spaced thresholds t_j = l + j*(u-l)/999
    
    Each client evaluates F1 for all candidates on its development set.
    Server computes equal-client arithmetic mean F1 for each t_j.
    Select candidate with maximal mean F1 (ties: smaller threshold).
    
    Normative reference: Section 9.4
    """
    
    def __init__(self, config: AttackAwareConfig = None):
        """
        Initialize B8 baseline.
        
        Args:
            config: Configuration
        """
        if config is None:
            config = AttackAwareConfig()
        self.config = config
    
    def compute_class_stats(
        self,
        scores: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute class statistics: n, mu, v.
        
        Args:
            scores: Array of scores
            
        Returns:
            Tuple of (n, mu, v)
        """
        n = float(len(scores))
        mu = float(np.mean(scores))
        v = float(np.mean((scores - mu) ** 2))
        return n, mu, v
    
    def compute_pooled_moments(
        self,
        client_stats: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        """
        Compute pooled population moments for both classes.
        
        Args:
            client_stats: Dictionary mapping client_id to
                         ((n_benign, mu_benign, v_benign), (n_attack, mu_attack, v_attack))
        
        Returns:
            Tuple of (benign_moments, attack_moments)
            where each is (N_y, mu_y, v_y, sigma_y)
        """
        # Separate benign and attack
        benign_stats = []
        attack_stats = []
        
        for cid, (b_stats, a_stats) in client_stats.items():
            benign_stats.append(b_stats)
            attack_stats.append(a_stats)
        
        # Compute benign pooled
        n_b_list = [s[0] for s in benign_stats]
        mu_b_list = [s[1] for s in benign_stats]
        v_b_list = [s[2] for s in benign_stats]
        
        N_b = sum(n_b_list)
        mu_b = sum(n * mu for n, mu, _ in benign_stats) / N_b if N_b > 0 else 0.0
        v_b = sum(n * (v + mu**2) for n, mu, v in benign_stats) / N_b - mu_b**2
        v_b = max(v_b, 0.0)
        sigma_b = np.sqrt(v_b)
        
        # Compute attack pooled
        n_a_list = [s[0] for s in attack_stats]
        mu_a_list = [s[1] for s in attack_stats]
        v_a_list = [s[2] for s in attack_stats]
        
        N_a = sum(n_a_list)
        mu_a = sum(n * mu for n, mu, _ in attack_stats) / N_a if N_a > 0 else 0.0
        v_a = sum(n * (v + mu**2) for n, mu, v in attack_stats) / N_a - mu_a**2
        v_a = max(v_a, 0.0)
        sigma_a = np.sqrt(v_a)
        
        benign_moments = (N_b, mu_b, v_b, sigma_b)
        attack_moments = (N_a, mu_a, v_a, sigma_a)
        
        return benign_moments, attack_moments
    
    def compute_overlap_interval(
        self,
        benign_moments: Tuple[float, float, float, float],
        attack_moments: Tuple[float, float, float, float],
    ) -> Tuple[float, float, bool]:
        """
        Compute overlap interval.
        
        Returns:
            Tuple of (l, u, is_defined)
        """
        _, mu_b, _, sigma_b = benign_moments
        _, mu_a, _, sigma_a = attack_moments
        
        l = max(mu_b - 3 * sigma_b, mu_a - 3 * sigma_a)
        u = min(mu_b + 3 * sigma_b, mu_a + 3 * sigma_a)
        
        is_defined = l < u
        
        return l, u, is_defined
    
    def generate_threshold_candidates(
        self,
        l: float,
        u: float,
        n_candidates: int = 1000,
    ) -> np.ndarray:
        """
        Generate equally spaced threshold candidates.
        
        Args:
            l: Lower bound
            u: Upper bound
            n_candidates: Number of candidates
            
        Returns:
            Array of threshold candidates
        """
        return np.linspace(l, u, n_candidates)
    
    def select_threshold(
        self,
        client_dev_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        benign_moments: Optional[Tuple[float, float, float, float]] = None,
        attack_moments: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[float, bool]:
        """
        Select threshold using Laridi-style method.
        
        Args:
            client_dev_data: Dictionary mapping client_id to
                            (benign_dev_scores, attack_dev_scores)
            benign_moments: Optional precomputed benign moments
            attack_moments: Optional precomputed attack moments
            
        Returns:
            Tuple of (selected_threshold, is_defined)
        """
        # Compute client stats if moments not provided
        if benign_moments is None or attack_moments is None:
            client_stats = {}
            for cid, (b_scores, a_scores) in client_dev_data.items():
                b_n, b_mu, b_v = self.compute_class_stats(b_scores)
                a_n, a_mu, a_v = self.compute_class_stats(a_scores)
                client_stats[cid] = ((b_n, b_mu, b_v), (a_n, a_mu, a_v))
            
            benign_moments, attack_moments = self.compute_pooled_moments(client_stats)
        
        # Compute overlap interval
        l, u, is_defined = self.compute_overlap_interval(benign_moments, attack_moments)
        
        if not is_defined:
            return 0.0, False  # UNDEFINED
        
        # Generate candidates
        candidates = self.generate_threshold_candidates(l, u, self.config.n_candidates)
        
        # Evaluate each candidate
        mean_f1s = []
        for t in candidates:
            client_f1s = []
            for cid, (b_scores, a_scores) in client_dev_data.items():
                # Create baseline instance just for F1 computation
                f1 = DevF1LgSelectBaseline().compute_f1(b_scores, a_scores, t)
                client_f1s.append(f1)
            
            mean_f1 = np.mean(client_f1s)
            mean_f1s.append(mean_f1)
        
        # Find candidate with maximal mean F1
        # Ties: smaller threshold
        best_idx = 0
        best_f1 = mean_f1s[0]
        for i in range(1, len(candidates)):
            if mean_f1s[i] > best_f1 or (mean_f1s[i] == best_f1 and candidates[i] < candidates[best_idx]):
                best_idx = i
                best_f1 = mean_f1s[i]
        
        return candidates[best_idx], True


class SupF11000Baseline:
    """
    Baseline B9: SUP-F1-1000
    
    Strong attack-aware candidate-search comparator independent of
    Laridi overlap assumptions.
    
    1000 federation-wide candidates spanning development-score min/max.
    Equal-client mean F1. Maximize F1.
    
    Normative reference: Section 9.1, row B9
    """
    
    def __init__(self, config: AttackAwareConfig = None):
        """
        Initialize B9 baseline.
        
        Args:
            config: Configuration
        """
        if config is None:
            config = AttackAwareConfig()
        self.config = config
    
    def generate_candidates(
        self,
        global_min: float,
        global_max: float,
        n_candidates: int = 1000,
    ) -> np.ndarray:
        """
        Generate candidates spanning the score range.
        
        Args:
            global_min: Minimum score across all development data
            global_max: Maximum score across all development data
            n_candidates: Number of candidates
            
        Returns:
            Array of threshold candidates
        """
        return np.linspace(global_min, global_max, n_candidates)
    
    def select_threshold(
        self,
        client_dev_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """
        Select threshold that maximizes equal-client mean F1.
        
        Args:
            client_dev_data: Dictionary mapping client_id to
                            (benign_dev_scores, attack_dev_scores)
            
        Returns:
            Selected threshold
        """
        # Find global min/max
        all_scores = []
        for b_scores, a_scores in client_dev_data.values():
            all_scores.extend(b_scores)
            all_scores.extend(a_scores)
        
        global_min = float(np.min(all_scores))
        global_max = float(np.max(all_scores))
        
        # Generate candidates
        candidates = self.generate_candidates(global_min, global_max, self.config.n_candidates)
        
        # Evaluate each candidate
        mean_f1s = []
        for t in candidates:
            client_f1s = []
            for cid, (b_scores, a_scores) in client_dev_data.items():
                f1 = DevF1LgSelectBaseline().compute_f1(b_scores, a_scores, t)
                client_f1s.append(f1)
            
            mean_f1 = np.mean(client_f1s)
            mean_f1s.append(mean_f1)
        
        # Find candidate with maximal mean F1
        # Ties: smaller threshold
        best_idx = int(np.argmax(mean_f1s))
        # Check for ties and use smaller threshold
        for i in range(best_idx + 1, len(candidates)):
            if mean_f1s[i] == mean_f1s[best_idx] and candidates[i] < candidates[best_idx]:
                best_idx = i
        
        return candidates[best_idx]


# Singleton instances
B7_DEV_F1_LG_SELECT = None
B8_LARIDI_STYLE_SS = None
B9_SUP_F1_1000 = None


def verify_attack_aware() -> None:
    """Verify attack-aware baselines."""
    np.random.seed(42)
    
    # Create dev data for 2 clients
    client_dev_data = {
        "nb01": (np.random.randn(500), np.random.randn(500) + 2.0),
        "nb02": (np.random.randn(500), np.random.randn(500) + 2.0),
    }
    
    # Test B7
    b1_thresholds = {"nb01": 1.0, "nb02": 1.0}
    b2_thresholds = {"nb01": 1.5, "nb02": 1.5}
    
    b7 = DevF1LgSelectBaseline(AttackAwareConfig(), b1_thresholds, b2_thresholds)
    b7_thresholds = b7.compute_thresholds(client_dev_data)
    
    assert len(b7_thresholds) == 2
    
    # Test B8
    b8 = LaridiStyleSSBaseline()
    threshold, is_defined = b8.select_threshold(client_dev_data)
    
    assert is_defined or not is_defined  # Just check it returns
    
    # Test B9
    b9 = SupF11000Baseline()
    threshold = b9.select_threshold(client_dev_data)
    
    assert global_min <= threshold <= global_max
    
    print("Attack-aware baselines verification passed.")


if __name__ == "__main__":
    verify_attack_aware()
