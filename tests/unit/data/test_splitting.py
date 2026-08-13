import numpy as np
import pandas as pd

from fedcrg.config.dataset_config import DatasetConfig, SplitConfig
from fedcrg.domain.enums import DataRole, DatasetFeatureContractId, DatasetId
from fedcrg.domain.identifiers import ClientId
from fedcrg.data.prepare import ClientData
from fedcrg.data.splits import DataSplitter


def _config() -> DatasetConfig:
    return DatasetConfig(
        id=DatasetId.NBAIOT,
        feature_contract=DatasetFeatureContractId.NBAIOT_LOCKED_115,
        source_version="1",
        feature_count=115,
        expected_clients=9,
        minimum_clients=1,
        expected_benign_counts={f"nb{i:02d}": 1 for i in range(1, 10)},
        split=SplitConfig(train_benign=10, reference_benign=5, mismatch_benign=8, calibration_benign=7, benign_guard=2, min_benign_test=5, attack_dev=6, min_attack_test=6, min_attack_test_per_group=2),
        calibration_seeds=(1000,),
        primary_calibration_seed=1000,
    )


def test_splits_are_disjoint_and_balanced() -> None:
    benign = pd.DataFrame({"f1": np.arange(40), "f2": np.arange(40)})
    attack = pd.DataFrame({"f1": np.arange(18), "f2": np.arange(18), "attack_group": ["a"] * 6 + ["b"] * 6 + ["c"] * 6})
    result = DataSplitter().split_base(ClientData(DatasetId.NBAIOT, ClientId("nb01"), benign, attack), _config())
    seen: set[str] = set()
    for frame in (item.frame for item in result.roles):
        ids = set(frame["row_id"])
        assert not seen.intersection(ids)
        seen.update(ids)
    dev = result.get(DataRole.ATTACK_DEV)
    assert len(dev) == 6
    assert dev["attack_group"].value_counts().to_dict() == {"a": 2, "b": 2, "c": 2}
