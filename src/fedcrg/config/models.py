"""Validated, immutable configuration models for reproducible experiment cells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from fedcrg.core.enums import (
    ActivationId,
    AggregationId,
    ComputeDeviceId,
    DatasetId,
    DeepSvddCenterMode,
    DetectorId,
    ExperimentId,
    OptimizerId,
    PolicyId,
    ProtocolId,
)
from fedcrg.core.types import OperatingBand


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FrozenModel(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}


class ProtocolConfig(FrozenModel):
    id: Literal[ProtocolId.FEDCRG] = ProtocolId.FEDCRG
    version: Literal["2.0"] = "2.0"
    alpha: float = Field(default=0.01, gt=0.0, lt=1.0)
    rho: float = Field(default=0.50, ge=0.0)
    readiness_assurance: float = Field(default=0.95, gt=0.0, lt=1.0)
    mismatch_confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    strict_exceedance: Literal[True] = True
    reject_calibration_ties: Literal[True] = True

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
    min_attack_test: int = Field(ge=0)
    min_attack_test_per_group: int = Field(gt=0)

    @property
    def reservoir_size(self) -> int:
        return (
            self.reference_benign
            + self.mismatch_benign
            + self.calibration_benign
            + self.benign_guard
        )

    @property
    def minimum_benign_rows(self) -> int:
        return self.train_benign + self.reservoir_size + self.min_benign_test


class DatasetConfig(FrozenModel):
    id: DatasetId
    source_version: str
    parser_version: str = "1"
    feature_count: int = Field(gt=0)
    expected_clients: int | None = Field(default=None, gt=0)
    expected_source_clients: int | None = Field(default=None, gt=0)
    minimum_clients: int = Field(default=1, gt=0)
    minimum_benign_rows: int | None = Field(default=None, gt=0)
    minimum_malicious_rows: int | None = Field(default=None, gt=0)
    split: SplitConfig
    calibration_seeds: tuple[int, ...]
    primary_calibration_seed: int
    expected_benign_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dataset_contract(self) -> "DatasetConfig":
        if not self.calibration_seeds:
            raise ValueError("At least one calibration seed is required")
        if len(set(self.calibration_seeds)) != len(self.calibration_seeds):
            raise ValueError("Calibration seeds must be unique")
        if self.primary_calibration_seed not in self.calibration_seeds:
            raise ValueError("Primary calibration seed must be in calibration_seeds")
        if self.id is DatasetId.NBAIOT:
            if self.feature_count != 115 or self.expected_clients != 9:
                raise ValueError("N-BaIoT contract requires 115 features and nine clients")
            if len(self.expected_benign_counts) != 9:
                raise ValueError("N-BaIoT requires the nine-client benign-count cross-check ledger")
        if self.id is DatasetId.DIAD:
            if self.feature_count != 86 or self.expected_source_clients != 105:
                raise ValueError("DIAD contract requires 86 features and 105 source identities")
            if self.minimum_benign_rows != 7800 or self.minimum_malicious_rows != 1000:
                raise ValueError("DIAD eligibility counts must remain 7800 benign / 1000 malicious")
            if self.minimum_clients != 10:
                raise ValueError("DIAD external validation requires at least ten eligible clients")
        return self


class AutoencoderConfig(FrozenModel):
    id: Literal[DetectorId.AUTOENCODER] = DetectorId.AUTOENCODER
    hidden_dims: tuple[int, ...]
    activation: Literal[ActivationId.TANH] = ActivationId.TANH
    xavier_tanh_gain: float = Field(default=5.0 / 3.0, gt=0.0)
    zero_bias: Literal[True] = True


class DeepSvddConfig(FrozenModel):
    id: Literal[DetectorId.DEEP_SVDD] = DetectorId.DEEP_SVDD
    hidden_dims: tuple[int, ...]
    embedding_dim: int = Field(gt=0)
    activation: Literal[ActivationId.TANH] = ActivationId.TANH
    bias: Literal[False] = False
    center_mode: Literal[
        DeepSvddCenterMode.EQUAL_MEAN_OF_CLIENT_INITIAL_EMBEDDINGS
    ] = DeepSvddCenterMode.EQUAL_MEAN_OF_CLIENT_INITIAL_EMBEDDINGS


DetectorConfig = Annotated[AutoencoderConfig | DeepSvddConfig, Field(discriminator="id")]


class TrainingConfig(FrozenModel):
    rounds: int = Field(default=30, gt=0)
    local_epochs: int = Field(gt=0)
    batch_size: int = Field(default=64, gt=0)
    optimizer: Literal[OptimizerId.ADAM] = OptimizerId.ADAM
    learning_rate_initial: float = Field(default=1e-3, gt=0.0)
    learning_rate_final: float = Field(default=1e-5, gt=0.0)
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = Field(default=1e-8, gt=0.0)
    weight_decay: Literal[0.0] = 0.0
    client_fraction: Literal[1.0] = 1.0
    aggregation: Literal[AggregationId.EQUAL_CLIENT_MEAN] = AggregationId.EQUAL_CLIENT_MEAN
    early_stopping: Literal[False] = False
    mixed_precision: Literal[False] = False
    deterministic_algorithms: Literal[True] = True
    record_round20_score_correlation: bool = False
    device: ComputeDeviceId = ComputeDeviceId.CPU

    @model_validator(mode="after")
    def validate_training_contract(self) -> "TrainingConfig":
        if self.learning_rate_final > self.learning_rate_initial:
            raise ValueError("Final learning rate cannot exceed initial learning rate")
        beta1, beta2 = self.adam_betas
        if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
            raise ValueError("Adam betas must be in (0, 1)")
        return self


class RandomnessConfig(FrozenModel):
    model_seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    attack_split_seed: Literal[9001] = 9001
    synthetic_seed: Literal[123456] = 123456
    bootstrap_seed: Literal[424242] = 424242

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
    def validate_experiment_contract(self) -> "ExperimentConfig":
        if not self.policies or len(set(self.policies)) != len(self.policies):
            raise ValueError("Policies must be non-empty and unique")
        if self.id is ExperimentId.SECOND_DETECTOR and self.detector.id is not DetectorId.DEEP_SVDD:
            raise ValueError("Second-detector experiment requires Deep-SVDD")
        if self.id is ExperimentId.EXTERNAL_DIAD and self.dataset.id is not DatasetId.DIAD:
            raise ValueError("External validation requires DIAD")
        return self

    def canonical_json(self) -> str:
        payload = self.model_dump(mode="json")
        payload["outputs_root"] = str(self.outputs_root)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def data_spec_hash(self) -> str:
        """Hash only inputs that determine the seed-independent prepared dataset cache."""
        dataset_payload = self.dataset.model_dump(mode="json", exclude={"calibration_seeds", "primary_calibration_seed"})
        return _sha256_json({
            "dataset": dataset_payload,
            "attack_split_seed": self.randomness.attack_split_seed,
        })

    @property
    def training_spec_hash(self) -> str:
        """Hash the exact detector-training specification, excluding policy/protocol axes."""
        return _sha256_json({
            "data_spec_hash": self.data_spec_hash,
            "detector": self.detector.model_dump(mode="json"),
            "training": self.training.model_dump(mode="json"),
        })
