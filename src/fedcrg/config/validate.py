"""Cross-component validation that cannot live inside one Pydantic model.

This module checks structural and relational invariants only. Scientific values are owned
by the YAML profiles. The frozen profile values themselves are locked by the contract
tests, not by this module.
"""

from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.domain.enums import (
    DatasetFeatureContractId,
    DatasetId,
    DetectorId,
    ExperimentId,
    PolicyId,
)
from fedcrg.domain.errors import ConfigurationError


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Validate frozen cross-model and training-profile invariants."""

    if config.dataset.id is not DatasetId.SYNTHETIC and not config.randomness.model_seeds:
        raise ConfigurationError("Real-data experiments require at least one model seed")

    if config.detector.id is DetectorId.DEEP_SVDD and config.training.local_epochs <= 0:
        raise ConfigurationError("Deep-SVDD requires positive local epochs")

    if config.dataset.id is DatasetId.DIAD and config.detector.id is DetectorId.AUTOENCODER:
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
    if len(registered) != 12:
        raise ConfigurationError("Policy catalogue must contain exactly 12 protocol policies")
