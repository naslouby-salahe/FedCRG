import json
from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.application.evaluate import EvaluatePolicies
from fedcrg.application.score import ComputeScores
from fedcrg.application.train import TrainDetector
from fedcrg.config.models import (
    AutoencoderConfig,
    DatasetConfig,
    ExperimentConfig,
    ProtocolConfig,
    RandomnessConfig,
    SplitConfig,
    TrainingConfig,
)
from fedcrg.core.enums import DataRole, DatasetId, ExperimentId, PolicyId
from fedcrg.scoring.cache import ScoreCache


def _config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        id=ExperimentId.PRIMARY_NBAIOT,
        protocol=ProtocolConfig(),
        dataset=DatasetConfig(
            id=DatasetId.NBAIOT,
            feature_count=2,
            expected_clients=2,
            minimum_clients=2,
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
        detector=AutoencoderConfig(hidden_dims=(2, 1)),
        training=TrainingConfig(rounds=1, local_epochs=1, batch_size=4, learning_rate_initial=1e-3, learning_rate_final=1e-3, client_fraction=1.0, device="cpu"),
        randomness=RandomnessConfig(model_seeds=(11,)),
        policies=(PolicyId.REFERENCE_QUANTILE, PolicyId.LOCAL_QUANTILE, PolicyId.FEDCRG),
        outputs_root=root,
    )


def _frame(rows: int, offset: float = 0.0) -> pd.DataFrame:
    x = np.linspace(0.0 + offset, 1.0 + offset, rows)
    return pd.DataFrame({"f1": x, "f2": x[::-1], "row_id": [f"r{i}-{offset}" for i in range(rows)]})


def _write_prepared(root: Path) -> Path:
    root.mkdir(parents=True)
    clients = {"c1": {}, "c2": {}}
    (root / "manifest.json").write_text(json.dumps({"clients": clients}), encoding="utf-8")
    role_rows = {DataRole.TRAIN: 4, DataRole.REFERENCE: 10, DataRole.MISMATCH: 736, DataRole.CALIBRATION: 1416, DataRole.BENIGN_GUARD: 10, DataRole.BENIGN_TEST: 20, DataRole.ATTACK_DEV: 10, DataRole.ATTACK_TEST: 20}
    for client_index, client_id in enumerate(clients):
        client_root = root / client_id
        client_root.mkdir()
        for role, rows in role_rows.items():
            offset = float(client_index)
            if role in {DataRole.ATTACK_DEV, DataRole.ATTACK_TEST}:
                offset += 4.0
            _frame(rows, offset).to_csv(client_root / f"{role.value}.csv.gz", index=False, compression="gzip")
    return root


def test_train_score_evaluate_cache_pipeline(tmp_path: Path) -> None:
    config = _config(tmp_path / "outputs")
    prepared = _write_prepared(tmp_path / "prepared")
    model_path, training_manifest = TrainDetector().train_from_cache(config, prepared, 11)
    assert model_path.exists()
    assert training_manifest.exists()
    score_root = ComputeScores().score_from_cache(config, prepared, model_path, 11)
    score_manifest = ScoreCache().load(score_root)
    assert set(score_manifest.clients) == {"c1", "c2"}
    evaluations = EvaluatePolicies().evaluate(config, score_manifest)
    assert len(evaluations) == 2 * len(config.policies)
    assert all(np.isfinite(item.threshold) for item in evaluations)
