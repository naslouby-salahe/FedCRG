"""
FedCRG Models Module

Provides detector model implementations for federated anomaly detection.

Normative reference: Section 8 (Frozen Detector and Federated Training)
"""

from fedcrg.models.base import BaseDetectorModel, ModelConfig
from fedcrg.models.autoencoder import (
    Autoencoder,
    AutoencoderConfig,
    compute_ae_param_count,
    create_autoencoder,
    create_nbaiot_ae_config,
    create_diad_ae_config,
    NBAIOT_ARCHITECTURE,
    DIAD_ARCHITECTURE,
    NBAIOT_PARAM_COUNT,
    DIAD_PARAM_COUNT,
)
from fedcrg.models.deep_svdd import (
    DeepSVDD,
    DeepSVDDConfig,
    compute_deep_svdd_param_count,
    create_deep_svdd,
    create_nbaiot_deep_svdd_config,
    DEEP_SVDD_ENCODER,
    DEEP_SVDD_PARAM_COUNT,
)

__all__ = [
    # Base classes
    "BaseDetectorModel",
    "ModelConfig",
    # Autoencoder
    "Autoencoder",
    "AutoencoderConfig",
    "compute_ae_param_count",
    "create_autoencoder",
    "create_nbaiot_ae_config",
    "create_diad_ae_config",
    "NBAIOT_ARCHITECTURE",
    "DIAD_ARCHITECTURE",
    "NBAIOT_PARAM_COUNT",
    "DIAD_PARAM_COUNT",
    # Deep-SVDD
    "DeepSVDD",
    "DeepSVDDConfig",
    "compute_deep_svdd_param_count",
    "create_deep_svdd",
    "create_nbaiot_deep_svdd_config",
    "DEEP_SVDD_ENCODER",
    "DEEP_SVDD_PARAM_COUNT",
]
