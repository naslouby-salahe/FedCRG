"""
Baseline Registry

Provides a registry of all baselines and factory functions.

Normative reference: Section 9
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

import numpy as np

from fedcrg.reference import Alpha
from fedcrg.baselines.quantile import (
    B0_REF_Q99_R,
    B1_GLOBAL_Q99_FULL,
    B2_LOCAL_Q99_FULL,
    B4_GATE_B_ONLY,
    QuantileBaseline,
    QuantileBaselineConfig,
)
from fedcrg.baselines.gate_only import B3_GATE_A_ONLY, GateAOnlyBaseline, GateAOnlyConfig
from fedcrg.baselines.shrinkage import B5_SHRINKAGE, ShrinkageBaseline, ShrinkageConfig
from fedcrg.baselines.feddetect_3sigma import B6_FEDDETECT_3SIGMA, FedDetect3SigmaBaseline
from fedcrg.baselines.attack_aware import (
    B7_DEV_F1_LG_SELECT,
    B8_LARIDI_STYLE_SS,
    B9_SUP_F1_1000,
    DevF1LgSelectBaseline,
    LaridiStyleSSBaseline,
    SupF11000Baseline,
    AttackAwareConfig,
)
from fedcrg.baselines.oracle import B10_ORACLE_TEST, OracleBaseline, OracleConfig


# Baseline registry mapping baseline_id to (class, config_class, is_benign_only)
BASELINE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "B0": {
        "name": "REF-Q99-R",
        "class": B0_REF_Q99_R,
        "config_class": QuantileBaselineConfig,
        "is_benign_only": True,
        "requires_attack": False,
        "requires_gate_a": False,
        "requires_gate_b": False,
        "description": "Reference threshold from R only",
    },
    "B1": {
        "name": "GLOBAL-Q99-FULL",
        "class": B1_GLOBAL_Q99_FULL,
        "config_class": QuantileBaselineConfig,
        "is_benign_only": True,
        "requires_attack": False,
        "requires_gate_a": False,
        "requires_gate_b": False,
        "description": "Global quantile from R+G+C",
    },
    "B2": {
        "name": "LOCAL-Q99-FULL",
        "class": B2_LOCAL_Q99_FULL,
        "config_class": QuantileBaselineConfig,
        "is_benign_only": True,
        "requires_attack": False,
        "requires_gate_a": False,
        "requires_gate_b": False,
        "description": "Local quantile from R+G+C",
    },
    "B3": {
        "name": "GATE-A-ONLY",
        "class": B3_GATE_A_ONLY,
        "config_class": GateAOnlyConfig,
        "is_benign_only": True,
        "requires_attack": False,
        "requires_gate_a": True,
        "requires_gate_b": False,
        "description": "Gate A only (ablates Gate B)",
    },
    "B4": {
        "name": "GATE-B-ONLY",
        "class": B4_GATE_B_ONLY,
        "config_class": QuantileBaselineConfig,
        "is_benign_only": True,
        "requires_attack": False,
        "requires_gate_a": False,
        "requires_gate_b": True,
        "description": "Gate B only (ablates Gate A readiness)",
    },
    "B5": {
        "name": "SHRINKAGE",
        "class": B5_SHRINKAGE,
        "config_class": ShrinkageConfig,
        "is_benign_only": True,
        "requires_attack": False,
        "requires_gate_a": False,
        "requires_gate_b": False,
        "description": "Shrinkage baseline",
    },
    "B6": {
        "name": "FEDDETECT-3SIGMA",
        "class": B6_FEDDETECT_3SIGMA,
        "config_class": None,
        "is_benign_only": True,
        "requires_attack": False,
        "requires_gate_a": False,
        "requires_gate_b": False,
        "description": "FedDetect-style 3-sigma threshold",
    },
    "B7": {
        "name": "DEV-F1-LG-SELECT",
        "class": DevF1LgSelectBaseline,
        "config_class": AttackAwareConfig,
        "is_benign_only": False,
        "requires_attack": True,
        "requires_gate_a": False,
        "requires_gate_b": False,
        "description": "Development F1 local-global selector",
    },
    "B8": {
        "name": "LARIDI-STYLE-SS",
        "class": LaridiStyleSSBaseline,
        "config_class": AttackAwareConfig,
        "is_benign_only": False,
        "requires_attack": True,
        "requires_gate_a": False,
        "requires_gate_b": False,
        "description": "Laridi-style summary-statistic overlap",
    },
    "B9": {
        "name": "SUP-F1-1000",
        "class": SupF11000Baseline,
        "config_class": AttackAwareConfig,
        "is_benign_only": False,
        "requires_attack": True,
        "requires_gate_a": False,
        "requires_gate_b": False,
        "description": "Supervised F1 with 1000 candidates",
    },
    "B10": {
        "name": "ORACLE-TEST",
        "class": OracleBaseline,
        "config_class": OracleConfig,
        "is_benign_only": False,
        "requires_attack": True,
        "requires_gate_a": False,
        "requires_gate_b": False,
        "description": "Oracle baseline (unattainable)",
    },
}


def get_baseline(baseline_id: str) -> Dict[str, Any]:
    """
    Get baseline information from registry.
    
    Args:
        baseline_id: Baseline identifier (e.g., "B0", "B1", ...)
        
    Returns:
        Dictionary with baseline information
    """
    if baseline_id not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {baseline_id}. Available: {list(BASELINE_REGISTRY.keys())}")
    
    return BASELINE_REGISTRY[baseline_id]


def get_all_baseline_ids() -> List[str]:
    """
    Get list of all registered baseline IDs.
    
    Returns:
        List of baseline IDs
    """
    return list(BASELINE_REGISTRY.keys())


def create_baseline(
    baseline_id: str,
    **kwargs,
) -> Any:
    """
    Create a baseline instance.
    
    Args:
        baseline_id: Baseline identifier
        **kwargs: Additional arguments for baseline constructor
        
    Returns:
        Baseline instance
    """
    info = get_baseline(baseline_id)
    baseline_class = info["class"]
    
    # Create instance
    return baseline_class(**kwargs)


def get_benign_only_baselines() -> List[str]:
    """
    Get list of benign-only baselines.
    
    Returns:
        List of baseline IDs that use only benign data
    """
    return [
        bid for bid, info in BASELINE_REGISTRY.items()
        if info["is_benign_only"]
    ]


def get_attack_aware_baselines() -> List[str]:
    """
    Get list of attack-aware baselines.
    
    Returns:
        List of baseline IDs that require attack data
    """
    return [
        bid for bid, info in BASELINE_REGISTRY.items()
        if info["requires_attack"]
    ]


def verify_registry() -> None:
    """Verify baseline registry."""
    # Check all baselines are registered
    expected_baselines = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"]
    registered = get_all_baseline_ids()
    
    for bid in expected_baselines:
        assert bid in registered, f"Baseline {bid} not registered"
    
    # Check benign-only classification
    benign_only = get_benign_only_baselines()
    assert "B0" in benign_only
    assert "B1" in benign_only
    assert "B2" in benign_only
    assert "B3" in benign_only
    assert "B4" in benign_only
    assert "B5" in benign_only
    assert "B6" in benign_only
    assert "B7" not in benign_only
    assert "B8" not in benign_only
    assert "B9" not in benign_only
    assert "B10" not in benign_only
    
    # Check attack-aware classification
    attack_aware = get_attack_aware_baselines()
    assert "B7" in attack_aware
    assert "B8" in attack_aware
    assert "B9" in attack_aware
    assert "B10" in attack_aware
    assert "B0" not in attack_aware
    
    # Test creating baselines
    b0 = create_baseline("B0", config=QuantileBaselineConfig())
    assert b0 is not None
    
    print("Baseline registry verification passed.")


if __name__ == "__main__":
    verify_registry()
