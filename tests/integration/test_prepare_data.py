from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.pipeline.prepare_dataset import PrepareData
from fedcrg.config.dataset_config import DatasetConfig, SplitConfig
from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.config.method_config import ProtocolConfig
from fedcrg.config.training_config import AutoencoderConfig, RandomnessConfig, TrainingConfig
from fedcrg.domain.enums import DatasetFeatureContractId, DatasetId, ExperimentId, PolicyId
from fedcrg.domain.identifiers import ClientId
from fedcrg.data.prepare import DatasetAdapter, ClientData

_NBAIOT_CLIENT_IDS = tuple(ClientId(f"nb{i:02d}") for i in range(1, 10))
_FEATURE_COLUMNS = [f"f{i}" for i in range(1, 116)]


class FakeAdapter(DatasetAdapter):
    @property
    def dataset_id(self) -> DatasetId:
        return DatasetId.NBAIOT

    def discover_clients(self) -> tuple[ClientId, ...]:
        return _NBAIOT_CLIENT_IDS

    def source_files(self) -> tuple[Path, ...]:
        return ()

    def load_client(self, client_id: ClientId) -> ClientData:
        offset = float(_NBAIOT_CLIENT_IDS.index(client_id))
        benign = pd.DataFrame(
            {
                "f1": np.arange(18, dtype=float) + offset,
                **{name: np.full(18, 5.0) for name in _FEATURE_COLUMNS[1:]},
            }
        )
        attack = pd.DataFrame(
            {
                "f1": np.arange(16, dtype=float) + 20.0,
                **{name: np.full(16, 5.0) for name in _FEATURE_COLUMNS[1:]},
                "attack_group": ["a"] * 8 + ["b"] * 8,
            }
        )
        return ClientData(DatasetId.NBAIOT, client_id, benign, attack)


class FakePrepare(PrepareData):
    @staticmethod
    def adapter(dataset: DatasetId, root: Path) -> DatasetAdapter:
        return FakeAdapter(root)


def _config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        id=ExperimentId.PRIMARY_NBAIOT,
        protocol=ProtocolConfig(),
        dataset=DatasetConfig(
            id=DatasetId.NBAIOT,
            feature_contract=DatasetFeatureContractId.NBAIOT_LOCKED_115,
            source_version="1",
            feature_count=115,
            expected_clients=9,
            minimum_clients=9,
            expected_benign_counts={client.value: 18 for client in _NBAIOT_CLIENT_IDS},
            split=SplitConfig(
                train_benign=4,
                reference_benign=2,
                mismatch_benign=2,
                calibration_benign=2,
                benign_guard=2,
                min_benign_test=4,
                attack_dev=4,
                min_attack_test=4,
                min_attack_test_per_group=2,
            ),
            calibration_seeds=(1000,),
            primary_calibration_seed=1000,
        ),
        detector=AutoencoderConfig(hidden_dims=(86, 57, 38, 29)),
        training=TrainingConfig(rounds=1, local_epochs=1),
        randomness=RandomnessConfig(model_seeds=(11,)),
        policies=(PolicyId.FEDCRG,),
        outputs_root=root,
    )


def test_prepare_data_writes_preprocessed_roles_and_evidence(tmp_path: Path) -> None:
    cache = FakePrepare().prepare(_config(tmp_path / "outputs"), tmp_path / "raw")
    assert (cache / "manifest.json").exists()
    assert (cache / "preprocessing.json").exists()
    assert (cache / "eligibility.json").exists()
    train = pd.read_csv(cache / "clients" / "nb01" / "train.csv.gz")
    assert train["f1"].between(0.0, 1.0).all()
    assert train["f2"].eq(0.0).all()
