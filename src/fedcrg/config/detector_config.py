"""Configuration for detector architectures and their frozen initialization rules."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from fedcrg.domain.enums import ActivationId, DeepSvddCenterMode, DetectorId


class AutoencoderConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid", "use_enum_values": False}

    id: Literal[DetectorId.AUTOENCODER] = DetectorId.AUTOENCODER
    hidden_dims: tuple[int, ...]
    activation: Literal[ActivationId.TANH] = ActivationId.TANH
    xavier_tanh_gain: float = Field(gt=0.0)
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
