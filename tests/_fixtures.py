"""Shared explicit test fixtures for the frozen primary scientific configuration.

These values mirror the frozen primary YAML profiles so unit tests never depend on
runtime resolution while also never declaring scientific defaults in production code.
They are regression fixtures for the locked primary contract.
"""

from __future__ import annotations

from fedcrg.configuration.detector_config import AutoencoderConfig
from fedcrg.configuration.method_config import ProtocolConfig
from fedcrg.configuration.statistics_config import StatisticsConfig
from fedcrg.configuration.training_config import RandomnessConfig, TrainingConfig
from fedcrg.domain.enums import (
    ActivationId,
    AggregationId,
    ComputeDeviceId,
    DetectorId,
    OptimizerId,
    ProtocolId,
)


def primary_protocol() -> ProtocolConfig:
    return ProtocolConfig(
        id=ProtocolId.FEDCRG,
        version="2.0",
        alpha=0.01,
        rho=0.50,
        readiness_assurance=0.95,
        mismatch_confidence=0.95,
        strict_exceedance=True,
        reject_calibration_ties=True,
    )


def primary_training() -> TrainingConfig:
    return TrainingConfig(
        rounds=30,
        local_epochs=120,
        batch_size=64,
        optimizer=OptimizerId.ADAM,
        learning_rate_initial=0.001,
        learning_rate_final=0.00001,
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        weight_decay=0.0,
        client_fraction=1.0,
        aggregation=AggregationId.EQUAL_CLIENT_MEAN,
        early_stopping=False,
        mixed_precision=False,
        deterministic_algorithms=True,
        record_round20_score_correlation=False,
        device=ComputeDeviceId.CUDA,
    )


def primary_randomness() -> RandomnessConfig:
    return RandomnessConfig(
        model_seeds=(11, 22, 33, 44, 55),
        attack_split_seed=9001,
        synthetic_seed=123456,
    )


def primary_statistics() -> StatisticsConfig:
    return StatisticsConfig(
        bootstrap_replicates=10000,
        bootstrap_seed=424242,
        utility_margin=0.03,
        utility_margin_sensitivities=(0.01, 0.05),
        familywise_alpha=0.05,
        ranking_invariance_tolerance=1e-12,
        shrinkage_n0_candidates=(100, 300, 1000, 3000, 10000),
        supervised_threshold_candidates=1000,
    )


def primary_autoencoder(hidden_dims: tuple[int, ...] = (86, 57, 38, 29)) -> AutoencoderConfig:
    return AutoencoderConfig(
        id=DetectorId.AUTOENCODER,
        hidden_dims=hidden_dims,
        activation=ActivationId.TANH,
        xavier_tanh_gain=5.0 / 3.0,
        zero_bias=True,
    )
