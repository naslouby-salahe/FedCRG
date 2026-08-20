from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from fedcrg.config import DatasetConfig, SplitConfig, Study
from fedcrg.data.datasets import ClientData
from fedcrg.data.splits import BaseSplitBuilder
from fedcrg.types import (
    ChronologyStatus,
    ClientId,
    DataIntegrityError,
    DataRole,
    DatasetId,
    ExperimentId,
)
from tests._fixtures import NBAIOT_CLIENT_IDS, nbaiot_dataset_config

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_DIAD_DATASET = Study.load().resolve(ExperimentId.EXTERNAL_DIAD).dataset


def _nbaiot_client() -> ClientData:
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
    result = BaseSplitBuilder().build(_nbaiot_client(), config, attack_split_seed=9001)
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


def test_nbaiot_split_rejects_insufficient_benign_rows() -> None:
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
        attack=_nbaiot_client().attack,
        chronology=ChronologyStatus.SOURCE_ORDER_ONLY,
    )
    builder = BaseSplitBuilder()
    with pytest.raises((ValueError, DataIntegrityError)):
        builder.build(client, config, attack_split_seed=9001)


def _diad_config() -> DatasetConfig:
    return _DIAD_DATASET.model_copy(
        update={
            "split": SplitConfig(
                train_benign=2,
                reference_benign=1,
                mismatch_benign=2,
                calibration_benign=2,
                benign_guard=1,
                min_benign_test=2,
                attack_dev=4,
                min_attack_test=4,
                min_attack_test_per_group=2,
            )
        }
    )


def test_diad_waterfill_preserves_small_categories_in_final_test() -> None:
    config = _diad_config()
    benign = pd.DataFrame({"f1": np.arange(10), "f2": np.arange(10)})
    attack = pd.DataFrame(
        {
            "f1": np.arange(10),
            "f2": np.arange(10),
            "attack_group": ["a"] * 3 + ["b"] * 7,
        }
    )
    result = BaseSplitBuilder().build(
        ClientData(
            dataset=DatasetId.DIAD,
            client_id=_CLIENT_ID_ADAPTER.validate_python("diad_example0001"),
            benign=benign,
            attack=attack,
            chronology=ChronologyStatus.SOURCE_ORDER_ONLY,
        ),
        config,
        attack_split_seed=9001,
    )
    dev = result.get(DataRole.ATTACK_DEV)
    test = result.get(DataRole.ATTACK_TEST)
    assert len(dev) == 4
    assert test["attack_group"].value_counts().to_dict()["a"] == 2
    assert len(test) == 6
    train = result.get(DataRole.TRAIN)
    assert len(train) == 2
    assert len(result.get(DataRole.BENIGN_TEST)) == 2
