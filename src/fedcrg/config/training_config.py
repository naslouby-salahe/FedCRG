"""Configuration for federated training and the randomness registry.

All scientific values are required and supplied by YAML profiles. The models declare no
scientific defaults.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from fedcrg.domain.enums import (
    AggregationId,
    ComputeDeviceId,
    OptimizerId,
)


class TrainingConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}

    rounds: int = Field(gt=0)
    local_epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    optimizer: Literal[OptimizerId.ADAM] = OptimizerId.ADAM
    learning_rate_initial: float = Field(gt=0.0)
    learning_rate_final: float = Field(gt=0.0)
    adam_betas: tuple[float, float]
    adam_epsilon: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0, le=0.0)
    client_fraction: float = Field(ge=1.0, le=1.0)
    aggregation: Literal[AggregationId.EQUAL_CLIENT_MEAN] = AggregationId.EQUAL_CLIENT_MEAN
    early_stopping: Literal[False] = False
    mixed_precision: Literal[False] = False
    deterministic_algorithms: Literal[True] = True
    record_round20_score_correlation: bool
    device: ComputeDeviceId

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

    model_seeds: tuple[int, ...]
    attack_split_seed: int
    synthetic_seed: int

    @model_validator(mode="after")
    def validate_model_seeds(self) -> RandomnessConfig:
        if len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("Model seeds must be unique")
        return self
