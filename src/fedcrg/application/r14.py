"""Canonical construction of the R14 DIAD feature-contract sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.models import AutoencoderConfig, ExperimentConfig
from fedcrg.core.enums import (
    ActivationId,
    DatasetFeatureContractId,
    DatasetId,
    ExperimentId,
    PolicyId,
)
from fedcrg.core.ids import ClientId
from fedcrg.data.feature_sensitivity import (
    NumericSafeFeatureContract,
    derive_numeric_safe_features,
)

_R14_POLICIES = (
    PolicyId.GLOBAL_QUANTILE,
    PolicyId.LOCAL_QUANTILE,
    PolicyId.SHRINKAGE,
    PolicyId.FEDCRG,
)


@dataclass(frozen=True, slots=True)
class R14Specification:
    config: ExperimentConfig
    feature_contract: NumericSafeFeatureContract
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class R14FeatureManifest:
    experiment_id: ExperimentId
    derivation_scope: str
    config_hash: str
    data_spec_hash: str
    training_spec_hash: str
    feature_contract_id: DatasetFeatureContractId
    feature_contract: NumericSafeFeatureContract


class R14FeatureSensitivityBuilder:
    """Freeze R14 from eligible-client training schema before outcome evidence opens."""

    def build(
        self,
        base_config: ExperimentConfig,
        training_frames: dict[ClientId, pd.DataFrame],
        manifest_path: Path,
    ) -> R14Specification:
        if base_config.dataset.id is not DatasetId.DIAD:
            raise ValueError("R14 can only be derived from the DIAD natural-client dataset")
        contract = derive_numeric_safe_features(training_frames)
        detector = AutoencoderConfig(
            hidden_dims=contract.encoder_hidden_dims,
            activation=ActivationId.TANH,
            xavier_tanh_gain=5.0 / 3.0,
            zero_bias=True,
        )
        payload = base_config.model_dump(mode="python")
        payload["id"] = ExperimentId.DIAD_FEATURE_SENSITIVITY
        dataset = dict(payload["dataset"])
        dataset["feature_contract"] = DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE
        dataset["feature_count"] = contract.dimension
        dataset["feature_names"] = contract.features
        payload["dataset"] = dataset
        payload["detector"] = detector.model_dump(mode="python")
        payload["policies"] = _R14_POLICIES
        config = ExperimentConfig.model_validate(payload)
        manifest = R14FeatureManifest(
            experiment_id=ExperimentId.DIAD_FEATURE_SENSITIVITY,
            derivation_scope="eligible_client_training_schema_only",
            config_hash=config.config_hash,
            data_spec_hash=config.data_spec_hash,
            training_spec_hash=config.training_spec_hash,
            feature_contract_id=DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE,
            feature_contract=contract,
        )
        atomic_write_json(manifest_path, manifest)
        return R14Specification(config, contract, manifest_path)
