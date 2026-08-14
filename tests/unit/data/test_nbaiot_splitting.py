from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fedcrg.data.datasets import ClientData
from fedcrg.data.splits import BaseSplitBuilder
from fedcrg.types import (
    ChronologyStatus,
    DataIntegrityError,
    DataRole,
    DatasetId,
)
from tests._fixtures import NBAIOT_CLIENT_IDS, nbaiot_dataset_config


def _client() -> ClientData:
    benign = pd.DataFrame(
        {
            "f1": np.arange(40, dtype=float),
            "f2": np.full(40, 5.0),
            "row_id": [f"{i:064x}" for i in range(40)],
        }
    )
    attack = pd.DataFrame(
        {
            "f1": np.arange(18, dtype=float) + 100.0,
            "f2": np.full(18, 5.0),
            "attack_group": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
            "row_id": [f"{i:064x}" for i in range(1000, 1018)],
        }
    )
    return ClientData(
        dataset=DatasetId.NBAIOT,
        client_id=NBAIOT_CLIENT_IDS[0],
        benign=benign,
        attack=attack,
        chronology=ChronologyStatus.SOURCE_ORDER_ONLY,
    )


def test_nbaiot_balanced_attack_development_allocation() -> None:
    config = nbaiot_dataset_config()
    result = BaseSplitBuilder().build(_client(), config, attack_split_seed=9001)
    dev = result.get(DataRole.ATTACK_DEV)
    assert dev["attack_group"].value_counts().to_dict() == {"a": 2, "b": 2, "c": 2}
    test = result.get(DataRole.ATTACK_TEST)
    assert test["attack_group"].value_counts().to_dict() == {"a": 4, "b": 4, "c": 4}
    dev_ids = set(dev["row_id"])
    test_ids = set(test["row_id"])
    assert dev_ids.isdisjoint(test_ids)
    train = result.get(DataRole.TRAIN)
    assert len(train) == 10
    assert train["label"].eq(0).all()
    assert dev["label"].eq(1).all()


def test_split_rejects_insufficient_benign_rows() -> None:
    config = nbaiot_dataset_config()
    benign = pd.DataFrame(
        {
            "f1": np.arange(5, dtype=float),
            "f2": np.full(5, 5.0),
            "row_id": [f"{i:064x}" for i in range(5)],
        }
    )
    client = ClientData(
        dataset=DatasetId.NBAIOT,
        client_id=NBAIOT_CLIENT_IDS[0],
        benign=benign,
        attack=_client().attack,
        chronology=ChronologyStatus.SOURCE_ORDER_ONLY,
    )
    with pytest.raises((ValueError, DataIntegrityError)):
        BaseSplitBuilder().build(client, config, attack_split_seed=9001)
