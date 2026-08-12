"""Cross-component configuration validation."""

from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import DatasetId, DetectorId
from fedcrg.core.exceptions import ConfigurationError


def validate_experiment_config(config: ExperimentConfig) -> None:
    if config.dataset.id is DatasetId.NBAIOT and config.dataset.feature_count != 115:
        raise ConfigurationError("N-BaIoT feature contract requires 115 features")
    if config.dataset.id is DatasetId.DIAD and config.dataset.feature_count != 86:
        raise ConfigurationError("DIAD primary feature contract requires 86 features")
    if config.detector.id is DetectorId.DEEP_SVDD and config.detector.embedding_dim <= 0:
        raise ConfigurationError("Deep-SVDD embedding dimension must be positive")
    if config.protocol.band.lower == 0.0:
        raise ConfigurationError("The current bidirectional mismatch protocol requires a positive lower band")
