"""
FedCRG Configuration Module

Implements typed configuration models using Pydantic for the FedCRG protocol.
This provides validation, serialization, and a single source of truth for all
runtime parameters.

Normative reference: Section 14.9 (Configuration validation) and Appendix E
(Normative Configuration Skeleton)
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from fedcrg.reference import (
    Alpha,
    A,
    B,
    GammaA,
    GammaB,
    Rho,
    PrimaryAlpha,
    PrimaryRho,
    PrimaryA,
    PrimaryB,
    PrimaryGammaA,
    PrimaryGammaB,
    compute_n_g_min,
)


# =============================================================================
# ENUMS FOR CONFIGURATION
# =============================================================================


class DatasetID(str, Enum):
    """Supported dataset identifiers."""

    NBAIOT = "nbaiot"
    DIAD = "diad"


class DetectorType(str, Enum):
    """Supported detector types."""

    AUTOENCODER = "autoencoder"
    DEEP_SVDD = "deep_svdd"


class AggregationType(str, Enum):
    """Federated aggregation types."""

    EQUAL_CLIENT_MEAN = "equal_client_mean"


class OptimizerType(str, Enum):
    """Supported optimizers."""

    ADAM = "adam"


class ActivationType(str, Enum):
    """Supported activation functions."""

    TANH = "tanh"
    LINEAR = "linear"


class PolicyID(str, Enum):
    """
    Threshold policy identifiers per Section 1168-1185.
    
    B0-B2: Benign-only baselines
    B3-B6: Ablations and published-style
    B7-B9: Attack-aware comparators
    B10: Oracle
    FEDCRG: The proposed method
    """

    REF_Q99_R = "REF-Q99-R"
    GLOBAL_Q99_FULL = "GLOBAL-Q99-FULL"
    LOCAL_Q99_FULL = "LOCAL-Q99-FULL"
    GATE_A_ONLY = "GATE-A-ONLY"
    GATE_B_ONLY = "GATE-B-ONLY"
    SHRINKAGE = "SHRINKAGE"
    FEDDETECT_3SIGMA = "FEDDETECT-3SIGMA"
    DEV_F1_LG_SELECT = "DEV-F1-LG-SELECT"
    LARIDI_STYLE_SS = "LARIDI-STYLE-SS"
    SUP_F1_1000 = "SUP-F1-1000"
    ORACLE_TEST = "ORACLE-TEST"
    FEDCRG = "FEDCRG"


# All policies list for validation
ALL_POLICIES = list(PolicyID)


# =============================================================================
# PROTOCOL CONFIGURATION
# =============================================================================


class ProtocolConfig(BaseModel):
    """
    Protocol-level configuration per Appendix E.
    
    These are the LOCKED values that define the FedCRG protocol.
    """

    id: str = Field(default="fedcrg", description="Method identifier")
    version: str = Field(default="2.0", description="Protocol version")
    
    # Core parameters
    alpha: float = Field(
        default=PrimaryAlpha(),
        description="Target benign false-positive rate"
    )
    rho: float = Field(
        default=PrimaryRho(),
        description="Relative practical tolerance around alpha"
    )
    gate_a_assurance: float = Field(
        default=PrimaryGammaA(),
        description="Gate-A in-band assurance level"
    )
    gate_b_confidence: float = Field(
        default=PrimaryGammaB(),
        description="Gate-B exact confidence level"
    )
    
    # Operational rules
    strict_threshold_operator: Literal[">"] = Field(
        default=">",
        description="Threshold comparison operator (strict greater-than)"
    )
    gate_b_min_mode: Literal["derived_from_a_gamma_b"] = Field(
        default="derived_from_a_gamma_b",
        description="How to compute n_G_min"
    )
    primary_gate_b_min_n_expected: int = Field(
        default=736,
        description="Expected primary Gate-B minimum (for verification)"
    )
    
    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        """Validate 0 < alpha < 1."""
        if not 0 < v < 1:
            raise ValueError(f"alpha must be in (0, 1), got {v}")
        return v
    
    @field_validator("rho")
    @classmethod
    def validate_rho(cls, v: float) -> float:
        """Validate rho >= 0."""
        if v < 0:
            raise ValueError(f"rho must be >= 0, got {v}")
        return v
    
    @field_validator("gate_a_assurance", "gate_b_confidence")
    @classmethod
    def validate_gamma(cls, v: float, info) -> float:
        """Validate 0 < gamma < 1."""
        if not 0 < v < 1:
            raise ValueError(f"{info.field_name} must be in (0, 1), got {v}")
        return v
    
    @model_validator(mode="after")
    def validate_derived_bands(self) -> "ProtocolConfig":
        """Validate derived bands a and b."""
        a = max(0.0, self.alpha * (1.0 - self.rho))
        b = min(1.0, self.alpha * (1.0 + self.rho))
        if not (0 <= a < b <= 1):
            raise ValueError(
                f"Derived bands invalid: a={a}, b={b}. Must have 0 <= a < b <= 1."
            )
        return self


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================


class TrainingConfig(BaseModel):
    """
    Federated training configuration per Section 8.1 and Appendix E.
    """

    model: DetectorType = Field(
        default=DetectorType.AUTOENCODER,
        description="Detector model type"
    )
    rounds: int = Field(
        default=30,
        ge=1,
        description="Number of federated rounds"
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Local batch size"
    )
    optimizer: OptimizerType = Field(
        default=OptimizerType.ADAM,
        description="Optimizer type"
    )
    
    # Adam hyperparameters
    lr_initial: float = Field(
        default=1e-3,
        description="Initial learning rate"
    )
    lr_final: float = Field(
        default=1e-5,
        description="Final learning rate"
    )
    betas: Tuple[float, float] = Field(
        default=(0.9, 0.999),
        description="Adam beta parameters"
    )
    eps: float = Field(
        default=1e-8,
        description="Adam epsilon"
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0,
        description="Weight decay"
    )
    
    # Training behavior
    client_fraction: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Fraction of clients participating each round"
    )
    aggregation: AggregationType = Field(
        default=AggregationType.EQUAL_CLIENT_MEAN,
        description="Parameter aggregation method"
    )
    early_stopping: bool = Field(
        default=False,
        description="Whether to use early stopping"
    )
    mixed_precision: bool = Field(
        default=False,
        description="Whether to use mixed precision"
    )
    
    # Detector-specific settings
    local_epochs_nbaiot: int = Field(
        default=120,
        ge=1,
        description="Local epochs per round for N-BaIoT"
    )
    local_epochs_diad: int = Field(
        default=20,
        ge=1,
        description="Local epochs per round for DIAD"
    )


# =============================================================================
# DEEP-SVDD CONFIGURATION
# =============================================================================


class DeepSVDDConfig(BaseModel):
    """
    Deep-SVDD specific configuration per Section 8.4.
    """

    rounds: int = Field(
        default=30,
        ge=1,
        description="Number of federated rounds"
    )
    local_epochs: int = Field(
        default=20,
        ge=1,
        description="Local epochs per round"
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Local batch size"
    )
    encoder: List[int] = Field(
        default=[115, 64, 32],
        description="Encoder layer sizes"
    )
    activation: ActivationType = Field(
        default=ActivationType.TANH,
        description="Hidden activation"
    )
    bias: bool = Field(
        default=False,
        description="Whether to use bias in encoder"
    )
    optimizer: OptimizerType = Field(
        default=OptimizerType.ADAM,
        description="Optimizer type"
    )
    lr_initial: float = Field(
        default=1e-3,
        description="Initial learning rate"
    )
    lr_final: float = Field(
        default=1e-5,
        description="Final learning rate"
    )
    center_mode: Literal["equal_mean_of_client_initial_embeddings"] = Field(
        default="equal_mean_of_client_initial_embeddings",
        description="Center computation mode"
    )


# =============================================================================
# N-BaIoT DATASET CONFIGURATION
# =============================================================================


class NBaiotConfig(BaseModel):
    """
    N-BaIoT dataset configuration per Section 7.1.
    """

    clients: int = Field(
        default=9,
        ge=1,
        description="Number of natural federated clients"
    )
    
    # Training data
    train_benign_per_client: int = Field(
        default=4000,
        ge=1,
        description="Benign training rows per client"
    )
    
    # Calibration reservoir
    reservoir_benign_per_client: int = Field(
        default=6000,
        ge=1,
        description="Benign reservoir rows per client"
    )
    
    # Role counts (from reservoir)
    reference_per_client: int = Field(
        default=500,
        ge=1,
        description="Reference sample count per client"
    )
    gate_per_client: int = Field(
        default=3000,
        ge=1,
        description="Gate sample count per client"
    )
    local_calibration_per_client: int = Field(
        default=2000,
        ge=1,
        description="Local calibration count per client"
    )
    comparator_benign_guard_per_client: int = Field(
        default=500,
        ge=1,
        description="Comparator benign guard count per client"
    )
    
    # Final test data
    min_final_benign_per_client: int = Field(
        default=3000,
        ge=1,
        description="Minimum final benign test rows per client"
    )
    
    # Attack data
    attack_dev_per_client: int = Field(
        default=500,
        ge=1,
        description="Attack development rows per client"
    )
    min_attack_test_rows_per_present_subtype: int = Field(
        default=100,
        ge=1,
        description="Minimum attack test rows per present subtype"
    )
    
    # Role assignment
    primary_calibration_seed: int = Field(
        default=1000,
        description="Primary calibration split seed"
    )
    calibration_seeds: List[int] = Field(
        default=list(range(1000, 1050)),
        description="All calibration split seeds"
    )


# =============================================================================
# DIAD DATASET CONFIGURATION
# =============================================================================


class DiadConfig(BaseModel):
    """
    CIC IoT-DIAD dataset configuration per Section 7.2.
    """

    # Eligibility thresholds
    min_benign_rows: int = Field(
        default=7800,
        ge=1,
        description="Minimum benign rows for eligibility"
    )
    min_malicious_rows: int = Field(
        default=1000,
        ge=1,
        description="Minimum malicious rows for eligibility"
    )
    min_final_attack_rows: int = Field(
        default=500,
        ge=1,
        description="Minimum final attack rows per client"
    )
    min_attack_test_rows_per_present_category: int = Field(
        default=100,
        ge=1,
        description="Minimum attack test rows per present category"
    )
    min_clients: int = Field(
        default=10,
        ge=1,
        description="Minimum number of eligible clients"
    )
    
    # Training data
    train_benign_per_client: int = Field(
        default=2000,
        ge=1,
        description="Benign training rows per client"
    )
    
    # Calibration reservoir
    reservoir_benign_per_client: int = Field(
        default=3800,
        ge=1,
        description="Benign reservoir rows per client"
    )
    
    # Role counts (from reservoir)
    reference_per_client: int = Field(
        default=300,
        ge=1,
        description="Reference sample count per client"
    )
    gate_per_client: int = Field(
        default=1500,
        ge=1,
        description="Gate sample count per client"
    )
    local_calibration_per_client: int = Field(
        default=1500,
        ge=1,
        description="Local calibration count per client"
    )
    comparator_benign_guard_per_client: int = Field(
        default=500,
        ge=1,
        description="Comparator benign guard count per client"
    )
    
    # Final test data
    min_final_benign_per_client: int = Field(
        default=2000,
        ge=1,
        description="Minimum final benign test rows per client"
    )
    
    # Attack data
    attack_dev_per_client: int = Field(
        default=500,
        ge=1,
        description="Attack development rows per client"
    )
    
    # Role assignment
    primary_calibration_seed: int = Field(
        default=2000,
        description="Primary calibration split seed"
    )
    calibration_seed_start: int = Field(
        default=2000,
        description="First calibration seed"
    )
    calibration_seed_end_inclusive: int = Field(
        default=2019,
        description="Last calibration seed (inclusive)"
    )


# =============================================================================
# RANDOMNESS CONFIGURATION
# =============================================================================


class RandomnessConfig(BaseModel):
    """
    Randomness configuration per Section 11.1.
    """

    model_seeds: List[int] = Field(
        default=[11, 22, 33, 44, 55],
        description="Primary detector model seeds"
    )
    attack_dev_seed: int = Field(
        default=9001,
        description="Attack development/test stratification seed"
    )
    synthetic_master_seed: int = Field(
        default=123456,
        description="Synthetic Monte Carlo master seed"
    )
    bootstrap_seed: int = Field(
        default=424242,
        description="Bootstrap seed"
    )


# =============================================================================
# POLICY CONFIGURATION
# =============================================================================


class PolicyConfig(BaseModel):
    """
    Policy registry configuration.
    """

    policies: List[PolicyID] = Field(
        default=ALL_POLICIES,
        description="List of threshold policies to evaluate"
    )
    
    @model_validator(mode="after")
    def validate_policies(self) -> "PolicyConfig":
        """Validate that all policies are in the allowed set."""
        for p in self.policies:
            if p not in ALL_POLICIES:
                raise ValueError(f"Unknown policy: {p}. Must be one of {ALL_POLICIES}")
        return self


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================


class FedCRGConfig(BaseModel):
    """
    Complete FedCRG configuration.
    
    This is the root configuration model that contains all sub-configurations.
    """

    protocol: ProtocolConfig = Field(
        default_factory=ProtocolConfig,
        description="Protocol-level configuration"
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration"
    )
    deep_svdd: DeepSVDDConfig = Field(
        default_factory=DeepSVDDConfig,
        description="Deep-SVDD configuration"
    )
    nbaiot: NBaiotConfig = Field(
        default_factory=NBaiotConfig,
        description="N-BaIoT dataset configuration"
    )
    diad: DiadConfig = Field(
        default_factory=DiadConfig,
        description="DIAD dataset configuration"
    )
    randomness: RandomnessConfig = Field(
        default_factory=RandomnessConfig,
        description="Randomness configuration"
    )
    policies: PolicyConfig = Field(
        default_factory=PolicyConfig,
        description="Policy registry"
    )
    
    @model_validator(mode="after")
    def validate_consistency(self) -> "FedCRGConfig":
        """Validate cross-field consistency."""
        # Validate that all seed lists contain unique integers
        for seed_list in [
            self.randomness.model_seeds,
            self.nbaiot.calibration_seeds,
        ]:
            if len(seed_list) != len(set(seed_list)):
                raise ValueError(f"Seed list contains duplicates: {seed_list}")
        
        # Validate N-BaIoT role counts fit in reservoir
        nbaiot_total_reservoir = (
            self.nbaiot.reference_per_client +
            self.nbaiot.gate_per_client +
            self.nbaiot.local_calibration_per_client +
            self.nbaiot.comparator_benign_guard_per_client
        )
        if nbaiot_total_reservoir != self.nbaiot.reservoir_benign_per_client:
            raise ValueError(
                f"N-BaIoT reservoir mismatch: roles sum to {nbaiot_total_reservoir}, "
                f"but reservoir_benign_per_client = {self.nbaiot.reservoir_benign_per_client}"
            )
        
        # Validate DIAD role counts fit in reservoir
        diad_total_reservoir = (
            self.diad.reference_per_client +
            self.diad.gate_per_client +
            self.diad.local_calibration_per_client +
            self.diad.comparator_benign_guard_per_client
        )
        if diad_total_reservoir != self.diad.reservoir_benign_per_client:
            raise ValueError(
                f"DIAD reservoir mismatch: roles sum to {diad_total_reservoir}, "
                f"but reservoir_benign_per_client = {self.diad.reservoir_benign_per_client}"
            )
        
        return self
    
    def get_nbaiot_calibration_seeds(self) -> List[int]:
        """Get N-BaIoT calibration seeds as a list."""
        return self.nbaiot.calibration_seeds
    
    def get_diad_calibration_seeds(self) -> List[int]:
        """Get DIAD calibration seeds as a list."""
        return list(range(
            self.diad.calibration_seed_start,
            self.diad.calibration_seed_end_inclusive + 1
        ))
    
    def get_model_seeds(self) -> List[int]:
        """Get model seeds for N-BaIoT and DIAD."""
        return self.randomness.model_seeds
    
    def get_deep_svdd_model_seeds(self) -> List[int]:
        """Get model seeds for Deep-SVDD."""
        return self.randomness.model_seeds[:3]  # [11, 22, 33]


# =============================================================================
# CONFIGURATION LOADING AND VALIDATION
# =============================================================================


def load_config(config_path: Path | str) -> FedCRGConfig:
    """
    Load a FedCRG configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Validated FedCRGConfig instance
    """
    import yaml
    
    config_path = Path(config_path)
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return FedCRGConfig(**config_dict)


def save_config(config: FedCRGConfig, config_path: Path | str) -> None:
    """
    Save a FedCRG configuration to a YAML file.
    
    Args:
        config: FedCRGConfig instance
        config_path: Path to save the YAML file
    """
    import yaml
    
    config_path = Path(config_path)
    config_dict = config.model_dump()
    
    # Convert enums and tuples to their string/list values for clean YAML serialization
    def convert_enum_to_str(obj):
        """Recursively convert enum instances and tuples to clean YAML types."""
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: convert_enum_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_enum_to_str(item) for item in obj]
        elif isinstance(obj, tuple):
            # Convert tuples to lists for clean YAML serialization
            return [convert_enum_to_str(item) for item in obj]
        else:
            return obj
    
    clean_dict = convert_enum_to_str(config_dict)
    
    with open(config_path, "w") as f:
        yaml.dump(clean_dict, f, default_flow_style=False, sort_keys=False)


def create_protocol_v2_config() -> FedCRGConfig:
    """
    Create the normative protocol_v2.yaml configuration.
    
    This matches Appendix E of the roadmap.
    """
    return FedCRGConfig(
        protocol=ProtocolConfig(
            id="fedcrg",
            version="2.0",
            alpha=0.01,
            rho=0.50,
            gate_a_assurance=0.95,
            gate_b_confidence=0.95,
            strict_threshold_operator=">",
            gate_b_min_mode="derived_from_a_gamma_b",
            primary_gate_b_min_n_expected=736,
        ),
        training=TrainingConfig(
            model=DetectorType.AUTOENCODER,
            rounds=30,
            local_epochs_nbaiot=120,
            local_epochs_diad=20,
            batch_size=64,
            optimizer=OptimizerType.ADAM,
            lr_initial=1e-3,
            lr_final=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            client_fraction=1.0,
            aggregation=AggregationType.EQUAL_CLIENT_MEAN,
            early_stopping=False,
            mixed_precision=False,
        ),
        deep_svdd=DeepSVDDConfig(
            rounds=30,
            local_epochs=20,
            batch_size=64,
            encoder=[115, 64, 32],
            activation=ActivationType.TANH,
            bias=False,
            optimizer=OptimizerType.ADAM,
            lr_initial=1e-3,
            lr_final=1e-5,
            center_mode="equal_mean_of_client_initial_embeddings",
        ),
        nbaiot=NBaiotConfig(
            clients=9,
            train_benign_per_client=4000,
            reservoir_benign_per_client=6000,
            reference_per_client=500,
            gate_per_client=3000,
            local_calibration_per_client=2000,
            comparator_benign_guard_per_client=500,
            min_final_benign_per_client=3000,
            attack_dev_per_client=500,
            min_attack_test_rows_per_present_subtype=100,
            primary_calibration_seed=1000,
            calibration_seeds=list(range(1000, 1050)),
        ),
        diad=DiadConfig(
            min_benign_rows=7800,
            min_malicious_rows=1000,
            min_final_attack_rows=500,
            min_attack_test_rows_per_present_category=100,
            min_clients=10,
            train_benign_per_client=2000,
            reservoir_benign_per_client=3800,
            reference_per_client=300,
            gate_per_client=1500,
            local_calibration_per_client=1500,
            comparator_benign_guard_per_client=500,
            min_final_benign_per_client=2000,
            attack_dev_per_client=500,
            primary_calibration_seed=2000,
            calibration_seed_start=2000,
            calibration_seed_end_inclusive=2019,
        ),
        randomness=RandomnessConfig(
            model_seeds=[11, 22, 33, 44, 55],
            attack_dev_seed=9001,
            synthetic_master_seed=123456,
            bootstrap_seed=424242,
        ),
        policies=PolicyConfig(
            policies=ALL_POLICIES,
        ),
    )


def create_nbaiot_primary_config() -> FedCRGConfig:
    """
    Create the N-BaIoT primary configuration.
    
    This is a convenience function for the N-BaIoT primary experiment.
    """
    config = create_protocol_v2_config()
    return config


def create_diad_external_config() -> FedCRGConfig:
    """
    Create the DIAD external validation configuration.
    
    This is a convenience function for the DIAD external experiment.
    """
    config = create_protocol_v2_config()
    return config


def create_synthetic_config() -> FedCRGConfig:
    """
    Create the synthetic experiment configuration.
    
    This is a convenience function for the synthetic Monte-Carlo experiments (S1-S6).
    It uses the same protocol parameters but may have different runtime settings
    for synthetic data generation.
    """
    return create_protocol_v2_config()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "DatasetID",
    "DetectorType",
    "AggregationType",
    "OptimizerType",
    "ActivationType",
    "PolicyID",
    "ALL_POLICIES",
    # Config classes
    "ProtocolConfig",
    "TrainingConfig",
    "DeepSVDDConfig",
    "NBaiotConfig",
    "DiadConfig",
    "RandomnessConfig",
    "PolicyConfig",
    "FedCRGConfig",
    # Functions
    "load_config",
    "save_config",
    "create_protocol_v2_config",
    "create_nbaiot_primary_config",
    "create_diad_external_config",
    "create_synthetic_config",
]
