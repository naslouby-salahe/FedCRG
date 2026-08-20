from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from fedcrg.config import (
    AutoencoderConfig,
    DatasetConfig,
    ExpectedBenignCounts,
    ExperimentConfig,
    ProtocolConfig,
    RandomnessConfig,
    SplitConfig,
    StatisticsConfig,
    SyntheticConfig,
    Study,
    TrainingConfig,
)
from fedcrg.data.datasets import hash_row_ids
from fedcrg.evidence.models import (
    ClientDatasetManifest,
    PreparedDatasetManifest,
    RoleArtifactManifest,
)
from fedcrg.evidence.store import PreparedDatasetManifestStore, atomic_write_json
from fedcrg.hashing import sha256_file
from fedcrg.types import (
    ActivationId,
    AggregationId,
    ClientId,
    ComputeDeviceId,
    DataRole,
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

_STUDY = Study.load()
_NBAIOT_DATASET = _STUDY.resolve(ExperimentId.PRIMARY_NBAIOT).dataset
_DIAD_DATASET = _STUDY.resolve(ExperimentId.EXTERNAL_DIAD).dataset

_DIAD_PIPELINE_FEATURES = ("f1", "f2", "f3", "f4")
_DIAD_PIPELINE_ROLE_ROWS = {
    DataRole.TRAIN: 4,
    DataRole.RESERVOIR: 2172,
    DataRole.BENIGN_TEST: 20,
    DataRole.ATTACK_DEV: 10,
    DataRole.ATTACK_TEST: 20,
}


def diad_pipeline_config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        id=ExperimentId.DIAD_FEATURE_SENSITIVITY,
        protocol=primary_protocol(),
        dataset=DatasetConfig(
            id=DatasetId.DIAD,
            feature_contract=DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE,
            source_version="1",
            parser_version="1",
            feature_count=4,
            feature_names=_DIAD_PIPELINE_FEATURES,
            diad_client_id_mac_digest_length=_DIAD_DATASET.diad_client_id_mac_digest_length,
            diad_finite_rate_minimum=_DIAD_DATASET.diad_finite_rate_minimum,
            expected_source_clients=115,
            minimum_clients=10,
            minimum_benign_rows=7800,
            minimum_malicious_rows=1000,
            expected_benign_counts=ExpectedBenignCounts({}),
            split=SplitConfig(
                train_benign=4,
                reference_benign=10,
                mismatch_benign=736,
                calibration_benign=1416,
                benign_guard=10,
                min_benign_test=20,
                attack_dev=10,
                min_attack_test=20,
                min_attack_test_per_group=5,
            ),
            calibration_seeds=(1000,),
            primary_calibration_seed=1000,
        ),
        detector=AutoencoderConfig(hidden_dims=(3, 2, 1, 1), xavier_tanh_gain=5.0 / 3.0),
        training=primary_training().model_copy(
            update={
                "rounds": 1,
                "local_epochs": 1,
                "batch_size": 4,
                "learning_rate_initial": 1e-3,
                "learning_rate_final": 1e-3,
                "device": ComputeDeviceId.CPU,
            }
        ),
        randomness=primary_randomness().model_copy(update={"model_seeds": (11,)}),
        statistics=primary_statistics(),
        synthetic=primary_synthetic(),
        policies=(PolicyId.REFERENCE_QUANTILE, PolicyId.LOCAL_QUANTILE, PolicyId.FEDCRG),
        outputs_root=root,
        preprocessed_root=root / "preprocessed",
    )


def _diad_row_id(client: str, role: DataRole, index: int) -> str:
    return hashlib.sha256(f"{client}-{role}-{index}".encode()).hexdigest()


def diad_role_frame(client: str, role: DataRole, rows: int, offset: float) -> pd.DataFrame:
    x = np.linspace(0.0 + offset, 1.0 + offset, rows)
    data: dict[str, object] = {
        name: x + index for index, name in enumerate(_DIAD_PIPELINE_FEATURES)
    }
    data["row_id"] = [str(_diad_row_id(client, role, i)) for i in range(rows)]
    if role in {DataRole.ATTACK_DEV, DataRole.ATTACK_TEST}:
        data["attack_group"] = ["atk"] * rows
    return pd.DataFrame(data)


def write_prepared_diad(root: Path, config: ExperimentConfig) -> Path:
    root.mkdir(parents=True)
    client_manifests: list[ClientDatasetManifest] = []
    for client_index, client_name in enumerate(("diad_test0001", "diad_test0002")):
        client_root = root / "clients" / client_name
        client_root.mkdir(parents=True)
        role_manifests: list[RoleArtifactManifest] = []
        for role, rows in _DIAD_PIPELINE_ROLE_ROWS.items():
            frame = diad_role_frame(client_name, role, rows, float(client_index))
            path = client_root / f"{role}.csv"
            frame.to_csv(path, index=False)
            role_manifests.append(
                RoleArtifactManifest(
                    role=role,
                    rows=rows,
                    row_id_sha256=hash_row_ids(frame["row_id"].tolist()),
                    relative_path=PurePosixPath(path.relative_to(root).as_posix()),
                    file_sha256=sha256_file(path),
                )
            )
        client_manifests.append(
            ClientDatasetManifest(client_id=client_name, roles=tuple(role_manifests))
        )

    manifest = PreparedDatasetManifest(
        dataset_id=config.dataset.id,
        source_version=config.dataset.source_version,
        parser_version=config.dataset.parser_version,
        data_spec_hash=config.data_spec_hash,
        feature_names=_DIAD_PIPELINE_FEATURES,
        clients=tuple(client_manifests),
        source_files=(),
        calibration_assignments=(),
        external_replication_supported=True,
        dataset_level_code=None,
        created_at=datetime.now(UTC),
    )
    PreparedDatasetManifestStore().save(root / "manifest.json", manifest)
    atomic_write_json(root / "preprocessing.json", {"note": "test fixture, values already scaled"})
    return root


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
        split_sensitivity_percentiles=(5, 25, 50, 75, 95),
        bootstrap_ci_percentiles=(2.5, 97.5),
        benchmark_p95_percentile=95,
        iqr_percentiles=(25, 75),
    )


def primary_synthetic() -> SyntheticConfig:
    return SyntheticConfig(
        lognormal_mean=0.0,
        lognormal_sigma=1.0,
        gamma_shape=2.0,
        gamma_scale=1.0,
        mixture_weight=0.1,
        mixture_shift=3.0,
        mixture_scale=1.0,
        coverage_tolerance_floor=0.005,
        coverage_tolerance_z=4.0,
        seed_decorrelation_scale=1000,
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
    return DatasetConfig(
        id=DatasetId.NBAIOT,
        feature_contract=DatasetFeatureContractId.NBAIOT_LOCKED_115,
        source_version="1",
        feature_count=115,
        feature_names=_NBAIOT_DATASET.feature_names,
        source_headers=_NBAIOT_DATASET.source_headers,
        device_directories=_NBAIOT_DATASET.device_directories,
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
        synthetic=primary_synthetic(),
        policies=(PolicyId.FEDCRG,),
        outputs_root=root,
        preprocessed_root=root / "preprocessed",
    )
