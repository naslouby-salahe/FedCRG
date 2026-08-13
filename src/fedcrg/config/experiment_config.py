"""Configuration for one complete, reproducible experiment cell."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, model_validator

from fedcrg.config.dataset_config import DatasetConfig
from fedcrg.config.detector_config import DetectorConfig
from fedcrg.config.method_config import ProtocolConfig
from fedcrg.config.statistics_config import StatisticsConfig
from fedcrg.config.training_config import RandomnessConfig, TrainingConfig
from fedcrg.domain.enums import (
    DatasetFeatureContractId,
    DatasetId,
    DetectorId,
    ExperimentId,
    PolicyId,
)


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class ExperimentConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}

    id: ExperimentId
    protocol: ProtocolConfig
    dataset: DatasetConfig
    detector: DetectorConfig
    training: TrainingConfig
    randomness: RandomnessConfig
    statistics: StatisticsConfig
    policies: tuple[PolicyId, ...]
    outputs_root: Path = Path("outputs")
    preprocessed_root: Path = Path("data/preprocessed")

    @model_validator(mode="after")
    def validate_experiment_contract(self) -> ExperimentConfig:
        if not self.policies or len(set(self.policies)) != len(self.policies):
            raise ValueError("Policies must be non-empty and unique")
        if self.id is ExperimentId.SECOND_DETECTOR and self.detector.id is not DetectorId.DEEP_SVDD:
            raise ValueError("Second-detector experiment requires Deep-SVDD")
        if self.id is ExperimentId.EXTERNAL_DIAD:
            if self.dataset.id is not DatasetId.DIAD:
                raise ValueError("External validation requires DIAD")
            if self.dataset.feature_contract is not DatasetFeatureContractId.DIAD_LOCKED_86:
                raise ValueError(
                    "Confirmatory DIAD external validation requires the locked 86-feature contract"
                )
        if self.id is ExperimentId.DIAD_FEATURE_SENSITIVITY:
            if self.dataset.id is not DatasetId.DIAD:
                raise ValueError("R14 requires DIAD")
            if (
                self.dataset.feature_contract
                is not DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE
            ):
                raise ValueError("R14 requires a frozen training-schema-derived feature contract")
        elif self.dataset.feature_contract is DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE:
            raise ValueError(
                "The training-schema-derived DIAD feature contract is exclusive to R14"
            )
        return self

    def serialized_payload(self) -> str:
        """Stable scientific configuration, independent of output location."""
        payload = self.model_dump(mode="json", exclude={"outputs_root", "preprocessed_root"})
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.serialized_payload().encode("utf-8")).hexdigest()

    @property
    def data_spec_hash(self) -> str:
        """Hash only inputs that determine the seed-independent prepared dataset cache."""
        dataset_payload = self.dataset.model_dump(
            mode="json",
            exclude={"calibration_seeds", "primary_calibration_seed"},
        )
        return _sha256_json(
            {
                "dataset": dataset_payload,
                "attack_split_seed": self.randomness.attack_split_seed,
            }
        )

    @property
    def training_spec_hash(self) -> str:
        """Hash the exact detector-training specification, excluding policy/protocol axes."""
        return _sha256_json(
            {
                "data_spec_hash": self.data_spec_hash,
                "detector": self.detector.model_dump(mode="json"),
                "training": self.training.model_dump(mode="json"),
            }
        )
