"""Smoke-test the real train -> score -> evaluate cache pipeline end-to-end."""

import hashlib
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from fedcrg.pipeline.evaluate_policies import EvaluatePolicies
from fedcrg.pipeline.compute_scores import ComputeScores
from fedcrg.pipeline.train_detector import TrainDetector
from fedcrg.artifacts.manifests import PreparedDatasetManifestStore
from fedcrg.artifacts.integrity import sha256_file
from fedcrg.artifacts.json_io import atomic_write_json
from fedcrg.config.dataset_config import DatasetConfig, SplitConfig
from fedcrg.config.detector_config import AutoencoderConfig
from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.domain.enums import (
    ComputeDeviceId,
    DataRole,
    DatasetFeatureContractId,
    DatasetId,
    ExperimentId,
    PolicyId,
)
from fedcrg.domain.identifiers import ClientId, RowId, Sha256
from fedcrg.data.prepare import ClientDatasetManifest, RoleArtifactManifest, hash_row_ids
from fedcrg.method.calibration_readiness import ReadinessPlanCache
from tests._fixtures import (
    primary_protocol,
    primary_randomness,
    primary_statistics,
    primary_training,
)

_FEATURES = ("f1", "f2", "f3", "f4")
_ROLE_ROWS = {
    DataRole.TRAIN: 4,
    DataRole.RESERVOIR: 2172,
    DataRole.BENIGN_TEST: 20,
    DataRole.ATTACK_DEV: 10,
    DataRole.ATTACK_TEST: 20,
}


def _config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        id=ExperimentId.DIAD_FEATURE_SENSITIVITY,
        protocol=primary_protocol(),
        dataset=DatasetConfig(
            id=DatasetId.DIAD,
            feature_contract=DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE,
            source_version="1",
            parser_version="1",
            feature_count=4,
            feature_names=_FEATURES,
            expected_source_clients=115,
            minimum_clients=10,
            minimum_benign_rows=7800,
            minimum_malicious_rows=1000,
            expected_benign_counts={},
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
        policies=(PolicyId.REFERENCE_QUANTILE, PolicyId.LOCAL_QUANTILE, PolicyId.FEDCRG),
        outputs_root=root,
    )


def _row_id(client: str, role: DataRole, index: int) -> RowId:
    return RowId(hashlib.sha256(f"{client}-{role.value}-{index}".encode()).hexdigest())


def _role_frame(client: str, role: DataRole, rows: int, offset: float) -> pd.DataFrame:
    x = np.linspace(0.0 + offset, 1.0 + offset, rows)
    data: dict[str, object] = {name: x + index for index, name in enumerate(_FEATURES)}
    data["row_id"] = [str(_row_id(client, role, i)) for i in range(rows)]
    if role in {DataRole.ATTACK_DEV, DataRole.ATTACK_TEST}:
        data["attack_group"] = ["atk"] * rows
    return pd.DataFrame(data)


def _write_prepared(root: Path, config: ExperimentConfig) -> Path:
    root.mkdir(parents=True)
    client_manifests: list[ClientDatasetManifest] = []
    for client_index, client_name in enumerate(("diad_test0001", "diad_test0002")):
        client_id = ClientId(client_name)
        client_root = root / "clients" / client_name
        client_root.mkdir(parents=True)
        role_manifests: list[RoleArtifactManifest] = []
        for role, rows in _ROLE_ROWS.items():
            frame = _role_frame(client_name, role, rows, float(client_index))
            path = client_root / f"{role.value}.csv"
            frame.to_csv(path, index=False)
            role_manifests.append(
                RoleArtifactManifest(
                    role=role,
                    rows=rows,
                    row_id_sha256=hash_row_ids(frame["row_id"].tolist()),
                    relative_path=PurePosixPath(path.relative_to(root).as_posix()),
                    file_sha256=Sha256(sha256_file(path)),
                )
            )
        client_manifests.append(ClientDatasetManifest(client_id, tuple(role_manifests)))

    manifest = PreparedDatasetManifestStore().build(
        dataset_id=config.dataset.id,
        source_version=config.dataset.source_version,
        parser_version=config.dataset.parser_version,
        data_spec_hash=Sha256(config.data_spec_hash),
        feature_names=_FEATURES,
        clients=tuple(client_manifests),
        source_files=(),
        calibration_assignments=(),
        external_replication_supported=True,
        dataset_level_code=None,
    )
    PreparedDatasetManifestStore().save(root / "manifest.json", manifest)
    atomic_write_json(root / "preprocessing.json", {"note": "test fixture, values already scaled"})
    return root


def test_train_score_evaluate_cache_pipeline(tmp_path: Path) -> None:
    config = _config(tmp_path / "outputs")
    prepared = _write_prepared(tmp_path / "prepared", config)

    model_path, training_manifest = TrainDetector().train_from_cache(config, prepared, 11)
    assert model_path.exists()
    assert training_manifest.exists()

    score_root = ComputeScores().score_from_cache(
        config, prepared, model_path, 11, training_manifest
    )

    readiness_cache_path = config.outputs_root / "cache" / "analysis" / "readiness_plans.json"
    ReadinessPlanCache(readiness_cache_path).precompute(
        config.dataset.split.calibration_benign,
        config.protocol.band,
        config.protocol.readiness_assurance,
    )

    bundle = EvaluatePolicies().evaluate_from_cache(config, score_root, calibration_seed=1000)
    assert len(bundle.clients) == 2 * len(config.policies)
    metrics = [item.metrics for item in bundle.clients if item.metrics is not None]
    assert metrics
    assert all(np.isfinite(item.fpr) for item in metrics)
