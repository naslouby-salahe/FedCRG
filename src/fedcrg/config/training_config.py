"""Configuration for detector architectures and federated training."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from fedcrg.domain.constants import (
    ATTACK_DEVELOPMENT_SEED,
    BOOTSTRAP_SEED,
    PRIMARY_MODEL_SEEDS,
    SYNTHETIC_MASTER_SEED,
)
from fedcrg.domain.enums import (
    ActivationId,
    AggregationId,
    ComputeDeviceId,
    DeepSvddCenterMode,
    DetectorId,
    OptimizerId,
)


class AutoencoderConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}

    id: Literal[DetectorId.AUTOENCODER] = DetectorId.AUTOENCODER
    hidden_dims: tuple[int, ...]
    activation: Literal[ActivationId.TANH] = ActivationId.TANH
    xavier_tanh_gain: float = Field(default=5.0 / 3.0, gt=0.0)
    zero_bias: Literal[True] = True


class DeepSvddConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}

    id: Literal[DetectorId.DEEP_SVDD] = DetectorId.DEEP_SVDD
    hidden_dims: tuple[int, ...]
    embedding_dim: int = Field(gt=0)
    activation: Literal[ActivationId.TANH] = ActivationId.TANH
    bias: Literal[False] = False
    center_mode: Literal[DeepSvddCenterMode.EQUAL_MEAN_OF_CLIENT_INITIAL_EMBEDDINGS] = (
        DeepSvddCenterMode.EQUAL_MEAN_OF_CLIENT_INITIAL_EMBEDDINGS
    )


DetectorConfig = Annotated[AutoencoderConfig | DeepSvddConfig, Field(discriminator="id")]


class TrainingConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}

    rounds: int = Field(default=30, gt=0)
    local_epochs: int = Field(gt=0)
    batch_size: int = Field(default=64, gt=0)
    optimizer: Literal[OptimizerId.ADAM] = OptimizerId.ADAM
    learning_rate_initial: float = Field(default=1e-3, gt=0.0)
    learning_rate_final: float = Field(default=1e-5, gt=0.0)
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = Field(default=1e-8, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0, le=0.0)
    client_fraction: float = Field(default=1.0, ge=1.0, le=1.0)
    aggregation: Literal[AggregationId.EQUAL_CLIENT_MEAN] = AggregationId.EQUAL_CLIENT_MEAN
    early_stopping: Literal[False] = False
    mixed_precision: Literal[False] = False
    deterministic_algorithms: Literal[True] = True
    record_round20_score_correlation: bool = False
    device: ComputeDeviceId = ComputeDeviceId.CPU

    @model_validator(mode="after")
    def validate_training_contract(self) -> TrainingConfig:
        if self.learning_rate_final > self.learning_rate_initial:
            raise ValueError("Final learning rate cannot exceed initial learning rate")
        beta1, beta2 = self.adam_betas
        if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
            raise ValueError("Adam betas must be in (0, 1)")
        return self


class RandomnessConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}

    model_seeds: tuple[int, ...] = PRIMARY_MODEL_SEEDS
    attack_split_seed: int = ATTACK_DEVELOPMENT_SEED
    synthetic_seed: int = SYNTHETIC_MASTER_SEED
    bootstrap_seed: int = BOOTSTRAP_SEED

    @model_validator(mode="after")
    def validate_model_seeds(self) -> RandomnessConfig:
        if not self.model_seeds or len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("Model seeds must be non-empty and unique")
        return self
