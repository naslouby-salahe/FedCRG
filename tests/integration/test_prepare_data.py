from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.application.prepare_data import PrepareData
from fedcrg.config.models import AutoencoderConfig, DatasetConfig, ExperimentConfig, ProtocolConfig, RandomnessConfig, SplitConfig, TrainingConfig
from fedcrg.core.enums import DatasetId, ExperimentId, PolicyId
from fedcrg.data.adapter import DatasetAdapter
from fedcrg.data.models import ClientData


class FakeAdapter(DatasetAdapter):
    @property
    def dataset_id(self) -> DatasetId:
        return DatasetId.NBAIOT

    def discover_clients(self) -> tuple[str, ...]:
        return ("c1", "c2")

    def load_client(self, client_id: str) -> ClientData:
        offset = 0.0 if client_id == "c1" else 10.0
        benign = pd.DataFrame({"f1": np.arange(18, dtype=float) + offset, "f2": np.full(18, 5.0)})
        attack = pd.DataFrame({"f1": np.arange(16, dtype=float) + 20.0, "f2": np.full(16, 5.0), "attack_group": ["a"] * 8 + ["b"] * 8})
        return ClientData(DatasetId.NBAIOT, client_id, benign, attack)


class FakePrepare(PrepareData):
    def adapter(self, dataset: DatasetId, root: Path) -> DatasetAdapter:
        return FakeAdapter(root)


def _config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(id=ExperimentId.PRIMARY_NBAIOT, protocol=ProtocolConfig(), dataset=DatasetConfig(id=DatasetId.NBAIOT, feature_count=2, expected_clients=2, minimum_clients=2, split=SplitConfig(train_benign=4, reference_benign=2, mismatch_benign=2, calibration_benign=2, benign_guard=2, min_benign_test=4, attack_dev=4, min_attack_test=4, min_attack_test_per_group=2), calibration_seeds=(1000,), primary_calibration_seed=1000), detector=AutoencoderConfig(hidden_dims=(1,)), training=TrainingConfig(rounds=1, local_epochs=1), randomness=RandomnessConfig(model_seeds=(11,)), policies=(PolicyId.FEDCRG,), outputs_root=root)


def test_prepare_data_writes_preprocessed_roles_and_evidence(tmp_path: Path) -> None:
    cache = FakePrepare().prepare(_config(tmp_path / "outputs"), tmp_path / "raw")
    assert (cache / "manifest.json").exists()
    assert (cache / "preprocessing.json").exists()
    assert (cache / "eligibility.json").exists()
    train = pd.read_csv(cache / "c1" / "train.csv.gz")
    assert train["f1"].between(0.0, 1.0).all()
    assert train["f2"].eq(0.0).all()
