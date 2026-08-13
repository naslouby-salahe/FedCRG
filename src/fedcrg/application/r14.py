"""Canonical construction of the R14 DIAD feature-contract sensitivity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.models import AutoencoderConfig, ExperimentConfig
from fedcrg.core.enums import ActivationId, DatasetId, ExperimentId, PolicyId
from fedcrg.core.ids import ClientId
from fedcrg.data.r14_feature_contract import R14FeatureContract, derive_r14_feature_contract

_R14_POLICIES = (
    PolicyId.GLOBAL_QUANTILE,
    PolicyId.LOCAL_QUANTILE,
    PolicyId.SHRINKAGE,
    PolicyId.FEDCRG,
)


@dataclass(frozen=True, slots=True)
class R14Specification:
    config: ExperimentConfig
    feature_contract: R14FeatureContract
    manifest_path: Path


class R14FeatureSensitivityBuilder:
    """Freeze R14 from eligible-client training schema before calibration/test outcomes."""

    def build(
        self,
        base_config: ExperimentConfig,
        training_frames: dict[ClientId, pd.DataFrame],
        manifest_path: Path,
    ) -> R14Specification:
        if base_config.dataset.id is not DatasetId.DIAD:
            raise ValueError("R14 can only be derived from the DIAD natural-client dataset")
        contract = derive_r14_feature_contract(training_frames)
        encoder_dims = contract.architecture[1:5]
        dataset = base_config.dataset.model_copy(update={"feature_count": contract.dimension})
        detector = AutoencoderConfig(
            hidden_dims=encoder_dims,
            activation=ActivationId.TANH,
            xavier_tanh_gain=5.0 / 3.0,
            zero_bias=True,
        )
        config = base_config.model_copy(
            update={
                "id": ExperimentId.DIAD_FEATURE_SENSITIVITY,
                "dataset": dataset,
                "detector": detector,
                "policies": _R14_POLICIES,
            }
        )
        atomic_write_json(
            manifest_path,
            {
                "experiment_id": ExperimentId.DIAD_FEATURE_SENSITIVITY.value,
                "derivation_scope": "eligible_client_training_schema_only",
                "config_hash": config.config_hash,
                "data_spec_hash": config.data_spec_hash,
                "training_spec_hash": config.training_spec_hash,
                **contract.to_dict(),
            },
        )
        return R14Specification(config, contract, manifest_path)
