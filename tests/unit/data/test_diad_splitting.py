import numpy as np
import pandas as pd

from fedcrg.config.models import DatasetConfig, SplitConfig
from fedcrg.core.constants import DIAD_EXPECTED_SOURCE_CLIENTS
from fedcrg.core.enums import DataRole, DatasetFeatureContractId, DatasetId
from fedcrg.core.ids import ClientId
from fedcrg.data.models import ClientData
from fedcrg.data.splitting import DataSplitter


def test_diad_waterfill_preserves_small_categories_in_final_test() -> None:
    config = DatasetConfig(
        id=DatasetId.DIAD,
        feature_contract=DatasetFeatureContractId.DIAD_LOCKED_86,
        source_version="1",
        feature_count=86,
        expected_source_clients=DIAD_EXPECTED_SOURCE_CLIENTS,
        minimum_clients=10,
        minimum_benign_rows=7800,
        minimum_malicious_rows=1000,
        split=SplitConfig(train_benign=2, reference_benign=1, mismatch_benign=2, calibration_benign=2, benign_guard=1, min_benign_test=2, attack_dev=4, min_attack_test=4, min_attack_test_per_group=2),
        calibration_seeds=(2000,),
        primary_calibration_seed=2000,
    )
    benign = pd.DataFrame({"f1": np.arange(10), "f2": np.arange(10)})
    attack = pd.DataFrame({"f1": np.arange(10), "f2": np.arange(10), "attack_group": ["a"] * 3 + ["b"] * 7})
    result = DataSplitter().split_base(ClientData(DatasetId.DIAD, ClientId("diad_example0001"), benign, attack), config)
    dev = result.get(DataRole.ATTACK_DEV)
    test = result.get(DataRole.ATTACK_TEST)
    assert len(dev) == 4
    assert test["attack_group"].value_counts().to_dict()["a"] == 2
    assert len(test) == 6
