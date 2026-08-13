import numpy as np
import pandas as pd

from fedcrg.config.models import DatasetConfig, SplitConfig
from fedcrg.core.enums import DataRole, DatasetId
from fedcrg.data.models import ClientData
from fedcrg.data.splitting import DataSplitter


def test_diad_waterfill_preserves_small_categories_in_final_test() -> None:
    config = DatasetConfig(id=DatasetId.DIAD, feature_count=2, minimum_clients=1, split=SplitConfig(train_benign=2, reference_benign=1, mismatch_benign=2, calibration_benign=2, benign_guard=1, min_benign_test=2, attack_dev=4, min_attack_test=4, min_attack_test_per_group=2), calibration_seeds=(2000,), primary_calibration_seed=2000)
    benign = pd.DataFrame({"f1": np.arange(10), "f2": np.arange(10)})
    attack = pd.DataFrame({"f1": np.arange(10), "f2": np.arange(10), "attack_group": ["a"] * 3 + ["b"] * 7})
    result = DataSplitter().split(ClientData(DatasetId.DIAD, "c1", benign, attack), config, 2000)
    dev = result.get(DataRole.ATTACK_DEV)
    test = result.get(DataRole.ATTACK_TEST)
    assert len(dev) == 4
    assert test["attack_group"].value_counts().to_dict()["a"] == 2
    assert len(test) == 6
