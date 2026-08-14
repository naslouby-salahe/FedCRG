"""Shared explicit test fixtures for the frozen primary scientific configuration.

These values mirror the frozen primary YAML profiles so unit tests never depend
on runtime resolution while also never declaring scientific defaults in
production code. They are regression fixtures for the locked primary contract.
"""

from __future__ import annotations

from pathlib import Path

from fedcrg.config import (
    AutoencoderConfig,
    DatasetConfig,
    ExpectedBenignCounts,
    ExperimentConfig,
    ProtocolConfig,
    RandomnessConfig,
    SplitConfig,
    StatisticsConfig,
    TrainingConfig,
)
from fedcrg.types import (
    ActivationId,
    AggregationId,
    ClientId,
    ComputeDeviceId,
    DatasetFeatureContractId,
    DatasetId,
    DetectorId,
    ExperimentId,
    OptimizerId,
    PolicyId,
    ProtocolId,
)
from pydantic import TypeAdapter

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)


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


NBAIOT_CLIENT_IDS = tuple(_CLIENT_ID_ADAPTER.validate_python(f"nb{i:02d}") for i in range(1, 10))


def nbaiot_dataset_config(
    *,
    train_benign: int = 10,
    reference_benign: int = 5,
    mismatch_benign: int = 8,
    calibration_benign: int = 7,
    benign_guard: int = 2,
    min_benign_test: int = 5,
    attack_dev: int = 6,
    min_attack_test: int = 6,
    min_attack_test_per_group: int = 2,
    expected_benign: int = 40,
) -> DatasetConfig:
    """Small N-BaIoT contract with configurable split sizing for unit tests."""
    return DatasetConfig(
        id=DatasetId.NBAIOT,
        feature_contract=DatasetFeatureContractId.NBAIOT_LOCKED_115,
        source_version="1",
        feature_count=115,
        expected_clients=9,
        minimum_clients=1,
        parser_version="1",
        expected_benign_counts=ExpectedBenignCounts(
            {client: expected_benign for client in NBAIOT_CLIENT_IDS}
        ),
        split=SplitConfig(
            train_benign=train_benign,
            reference_benign=reference_benign,
            mismatch_benign=mismatch_benign,
            calibration_benign=calibration_benign,
            benign_guard=benign_guard,
            min_benign_test=min_benign_test,
            attack_dev=attack_dev,
            min_attack_test=min_attack_test,
            min_attack_test_per_group=min_attack_test_per_group,
        ),
        calibration_seeds=(1000,),
        primary_calibration_seed=1000,
    )


def primary_experiment_config(
    root: Path, experiment_id: ExperimentId = ExperimentId.PRIMARY_NBAIOT
) -> ExperimentConfig:
    """Resolved primary-shaped experiment configuration rooted at ``root``."""
    return ExperimentConfig(
        id=experiment_id,
        protocol=primary_protocol(),
        dataset=nbaiot_dataset_config(),
        detector=primary_autoencoder(),
        training=primary_training().model_copy(
            update={"rounds": 1, "local_epochs": 1, "batch_size": 2, "device": ComputeDeviceId.CPU}
        ),
        randomness=primary_randomness(),
        statistics=primary_statistics(),
        policies=(PolicyId.FEDCRG,),
        outputs_root=root,
        preprocessed_root=root / "preprocessed",
    )
