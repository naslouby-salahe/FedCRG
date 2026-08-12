"""
FedCRG Baseline Suite

Implements all baselines from Section 9.

Normative reference: Section 9
"""

# Quantile baselines (B0-B2, B4)
from fedcrg.baselines.quantile import (
    QuantileBaseline,
    QuantileBaselineConfig,
    B0_REF_Q99_R,
    B1_GLOBAL_Q99_FULL,
    B2_LOCAL_Q99_FULL,
    B4_GATE_B_ONLY,
    compute_quantile_threshold,
)

# Gate baselines (B3)
from fedcrg.baselines.gate_only import (
    B3_GATE_A_ONLY,
    GateAOnlyBaseline,
    GateAOnlyConfig,
)

# Shrinkage baseline (B5)
from fedcrg.baselines.shrinkage import (
    B5_SHRINKAGE,
    ShrinkageBaseline,
    ShrinkageConfig,
    select_best_n0,
)

# FedDetect 3-sigma baseline (B6)
from fedcrg.baselines.feddetect_3sigma import (
    B6_FEDDETECT_3SIGMA,
    FedDetect3SigmaBaseline,
)

# Attack-aware baselines (B7-B9)
from fedcrg.baselines.attack_aware import (
    B7_DEV_F1_LG_SELECT,
    B8_LARIDI_STYLE_SS,
    B9_SUP_F1_1000,
    DevF1LgSelectBaseline,
    LaridiStyleSSBaseline,
    SupF11000Baseline,
    AttackAwareConfig,
)

# Oracle baseline (B10)
from fedcrg.baselines.oracle import (
    B10_ORACLE_TEST,
    OracleBaseline,
    OracleConfig,
)

# Registry and factory
from fedcrg.baselines.registry import (
    BASELINE_REGISTRY,
    get_baseline,
    get_all_baseline_ids,
    create_baseline,
)

__all__ = [
    # Quantile baselines
    "QuantileBaseline",
    "QuantileBaselineConfig",
    "B0_REF_Q99_R",
    "B1_GLOBAL_Q99_FULL",
    "B2_LOCAL_Q99_FULL",
    "B4_GATE_B_ONLY",
    "compute_quantile_threshold",
    # Gate only
    "B3_GATE_A_ONLY",
    "GateAOnlyBaseline",
    "GateAOnlyConfig",
    # Shrinkage
    "B5_SHRINKAGE",
    "ShrinkageBaseline",
    "ShrinkageConfig",
    "select_best_n0",
    # FedDetect 3-sigma
    "B6_FEDDETECT_3SIGMA",
    "FedDetect3SigmaBaseline",
    # Attack-aware
    "B7_DEV_F1_LG_SELECT",
    "B8_LARIDI_STYLE_SS",
    "B9_SUP_F1_1000",
    "DevF1LgSelectBaseline",
    "LaridiStyleSSBaseline",
    "SupF11000Baseline",
    "AttackAwareConfig",
    # Oracle
    "B10_ORACLE_TEST",
    "OracleBaseline",
    "OracleConfig",
    # Registry
    "BASELINE_REGISTRY",
    "get_baseline",
    "get_all_baseline_ids",
    "create_baseline",
]
