"""Cross-component validation for frozen experiment configurations."""

from fedcrg.config.models import ExperimentConfig
from fedcrg.core.constants import PRIMARY_MODEL_SEEDS
from fedcrg.core.enums import DatasetId, DetectorId, PolicyId
from fedcrg.core.exceptions import ConfigurationError
from fedcrg.policies.registry import PolicyRegistry


def validate_experiment_config(config: ExperimentConfig) -> None:
    if config.dataset.id is DatasetId.NBAIOT:
        if config.dataset.feature_count != 115:
            raise ConfigurationError("N-BaIoT feature contract requires 115 features")
        if config.detector.id is DetectorId.AUTOENCODER and config.training.local_epochs != 120:
            raise ConfigurationError("N-BaIoT primary autoencoder requires 120 local epochs")
    if config.dataset.id is DatasetId.DIAD:
        if config.dataset.feature_count != 86:
            raise ConfigurationError("DIAD confirmatory feature contract requires 86 features")
        if config.detector.id is DetectorId.AUTOENCODER and config.training.local_epochs != 20:
            raise ConfigurationError("DIAD confirmatory autoencoder requires 20 local epochs")
    if config.detector.id is DetectorId.DEEP_SVDD:
        if config.detector.embedding_dim != 32:
            raise ConfigurationError("Deep-SVDD sensitivity requires a 32-dimensional embedding")
        if config.training.local_epochs != 20:
            raise ConfigurationError("Deep-SVDD sensitivity requires 20 local epochs")
    if config.dataset.id is not DatasetId.SYNTHETIC and config.training.rounds != 30:
        raise ConfigurationError("Confirmatory detector training requires exactly 30 rounds")
    if config.dataset.id is DatasetId.NBAIOT and config.detector.id is DetectorId.AUTOENCODER:
        if config.randomness.model_seeds != PRIMARY_MODEL_SEEDS:
            raise ConfigurationError("Primary model seeds must remain 11,22,33,44,55")
    if any(policy not in set(PolicyId) for policy in config.policies):
        raise ConfigurationError("Experiment contains an unregistered policy")
    PolicyRegistry().assert_exact_protocol_registry()
