"""Validated configuration models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from fedcrg.core.enums import (
    ActivationId, AggregationId, ComputeDeviceId, DatasetId, DetectorId,
    ExperimentId, OptimizerId, PolicyId,
)
from fedcrg.core.types import OperatingBand


class FrozenModel(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}


class ProtocolConfig(FrozenModel):
    version: str = "2.0"
    alpha: float = Field(default=0.01, gt=0.0, lt=1.0)
    rho: float = Field(default=0.50, ge=0.0)
    readiness_assurance: float = Field(default=0.95, gt=0.0, lt=1.0)
    mismatch_confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    strict_exceedance: Literal[True] = True
    reject_calibration_ties: bool = True

    @property
    def band(self) -> OperatingBand:
        return OperatingBand(
            lower=max(0.0, self.alpha * (1.0 - self.rho)),
            upper=min(1.0, self.alpha * (1.0 + self.rho)),
        )


class SplitConfig(FrozenModel):
    train_benign: int = Field(gt=0)
    reference_benign: int = Field(gt=0)
    mismatch_benign: int = Field(gt=0)
    calibration_benign: int = Field(gt=0)
    benign_guard: int = Field(ge=0)
    min_benign_test: int = Field(gt=0)
    attack_dev: int = Field(gt=0)
    min_attack_test: int = Field(gt=0)
    min_attack_test_per_group: int = Field(gt=0)

    @property
    def reservoir_size(self) -> int:
        return (
            self.reference_benign
            + self.mismatch_benign
            + self.calibration_benign
            + self.benign_guard
        )


class DatasetConfig(FrozenModel):
    id: DatasetId
    feature_count: int = Field(gt=0)
    expected_clients: int | None = Field(default=None, gt=0)
    minimum_clients: int = Field(default=1, gt=0)
    split: SplitConfig
    calibration_seeds: tuple[int, ...]
    primary_calibration_seed: int

    @model_validator(mode="after")
    def validate_seeds(self) -> "DatasetConfig":
        if not self.calibration_seeds:
            raise ValueError("At least one calibration seed is required")
        if len(set(self.calibration_seeds)) != len(self.calibration_seeds):
            raise ValueError("Calibration seeds must be unique")
        if self.primary_calibration_seed not in self.calibration_seeds:
            raise ValueError("Primary calibration seed must be in calibration_seeds")
        return self


class AutoencoderConfig(FrozenModel):
    id: Literal[DetectorId.AUTOENCODER] = DetectorId.AUTOENCODER
    hidden_dims: tuple[int, ...]
    activation: ActivationId = ActivationId.TANH


class DeepSvddConfig(FrozenModel):
    id: Literal[DetectorId.DEEP_SVDD] = DetectorId.DEEP_SVDD
    hidden_dims: tuple[int, ...]
    embedding_dim: int = Field(gt=0)
    activation: ActivationId = ActivationId.TANH
    bias: bool = False


DetectorConfig = Annotated[AutoencoderConfig | DeepSvddConfig, Field(discriminator="id")]


class TrainingConfig(FrozenModel):
    rounds: int = Field(default=30, gt=0)
    local_epochs: int = Field(gt=0)
    batch_size: int = Field(default=64, gt=0)
    optimizer: OptimizerId = OptimizerId.ADAM
    learning_rate_initial: float = Field(default=1e-3, gt=0.0)
    learning_rate_final: float = Field(default=1e-5, gt=0.0)
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = Field(default=1e-8, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    client_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    aggregation: AggregationId = AggregationId.EQUAL_CLIENT_MEAN
    mixed_precision: bool = False
    device: ComputeDeviceId = ComputeDeviceId.CPU

    @model_validator(mode="after")
    def validate_learning_rates(self) -> "TrainingConfig":
        if self.learning_rate_final > self.learning_rate_initial:
            raise ValueError("Final learning rate cannot exceed initial learning rate")
        b1, b2 = self.adam_betas
        if not 0.0 < b1 < 1.0 or not 0.0 < b2 < 1.0:
            raise ValueError("Adam betas must be in (0, 1)")
        return self


class RandomnessConfig(FrozenModel):
    model_seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    attack_split_seed: int = 9001
    synthetic_seed: int = 123456
    bootstrap_seed: int = 424242

    @model_validator(mode="after")
    def validate_model_seeds(self) -> "RandomnessConfig":
        if not self.model_seeds or len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("Model seeds must be non-empty and unique")
        return self


class ExperimentConfig(FrozenModel):
    id: ExperimentId
    protocol: ProtocolConfig
    dataset: DatasetConfig
    detector: DetectorConfig
    training: TrainingConfig
    randomness: RandomnessConfig = RandomnessConfig()
    policies: tuple[PolicyId, ...]
    outputs_root: Path = Path("outputs")

    @model_validator(mode="after")
    def validate_policies(self) -> "ExperimentConfig":
        if not self.policies or len(set(self.policies)) != len(self.policies):
            raise ValueError("Policies must be non-empty and unique")
        return self

    def canonical_json(self) -> str:
        payload = self.model_dump(mode="json")
        payload["outputs_root"] = str(self.outputs_root)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
