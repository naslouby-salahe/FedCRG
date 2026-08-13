"""Cross-component validation that cannot live inside one Pydantic model."""

from fedcrg.config.models import ExperimentConfig
from fedcrg.core.constants import DEEP_SVDD_MODEL_SEEDS, PRIMARY_MODEL_SEEDS
from fedcrg.core.enums import (
    DatasetFeatureContractId,
    DatasetId,
    DetectorId,
    ExperimentId,
    PolicyId,
)
from fedcrg.core.exceptions import ConfigurationError
from fedcrg.policies.registry import PolicyRegistry


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Validate frozen cross-registry and training-profile invariants.

    Dataset-schema invariants are owned by ``DatasetConfig``. This function only
    checks relationships spanning detector/training/randomness/policy registries.
    """

    if config.dataset.id is not DatasetId.SYNTHETIC and config.training.rounds != 30:
        raise ConfigurationError("Real-data detector training requires exactly 30 rounds")

    if config.detector.id is DetectorId.DEEP_SVDD:
        if config.detector.embedding_dim != 32 or config.training.local_epochs != 20:
            raise ConfigurationError(
                "Deep-SVDD sensitivity requires a 32-dimensional embedding and 20 local epochs"
            )
        if config.randomness.model_seeds != DEEP_SVDD_MODEL_SEEDS:
            raise ConfigurationError("Deep-SVDD model seeds must remain 11,22,33")

    if config.dataset.id is DatasetId.NBAIOT and config.detector.id is DetectorId.AUTOENCODER:
        if config.training.local_epochs != 120:
            raise ConfigurationError("N-BaIoT primary autoencoder requires 120 local epochs")
        if config.randomness.model_seeds != PRIMARY_MODEL_SEEDS:
            raise ConfigurationError("Primary model seeds must remain 11,22,33,44,55")

    if config.dataset.id is DatasetId.DIAD and config.detector.id is DetectorId.AUTOENCODER:
        if config.training.local_epochs != 20:
            raise ConfigurationError("DIAD autoencoder experiments require 20 local epochs")
        if config.id is ExperimentId.DIAD_FEATURE_SENSITIVITY:
            if (
                config.dataset.feature_contract
                is not DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE
            ):
                raise ConfigurationError("R14 must use the training-derived DIAD feature contract")
        elif config.dataset.feature_contract is not DatasetFeatureContractId.DIAD_LOCKED_86:
            raise ConfigurationError(
                "Confirmatory DIAD experiments require the locked 86-feature contract"
            )

    registered = set(PolicyId)
    if any(policy not in registered for policy in config.policies):
        raise ConfigurationError("Experiment contains an unregistered policy")
    PolicyRegistry().assert_exact_protocol_registry()
