"""Unit tests for deterministic base-role splitting (DIAD water-filling)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from fedcrg.config import DatasetConfig, ExpectedBenignCounts, SplitConfig
from fedcrg.data.datasets import ClientData
from fedcrg.data.splits import BaseSplitBuilder
from fedcrg.types import (
    ChronologyStatus,
    ClientId,
    DataRole,
    DatasetFeatureContractId,
    DatasetId,
)

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)


def _diad_config() -> DatasetConfig:
    return DatasetConfig(
        id=DatasetId.DIAD,
        feature_contract=DatasetFeatureContractId.DIAD_LOCKED_86,
        source_version="1",
        parser_version="1",
        feature_count=86,
        expected_source_clients=115,
        minimum_clients=10,
        minimum_benign_rows=7800,
        minimum_malicious_rows=1000,
        expected_benign_counts=ExpectedBenignCounts({}),
        split=SplitConfig(
            train_benign=2,
            reference_benign=1,
            mismatch_benign=2,
            calibration_benign=2,
            benign_guard=1,
            min_benign_test=2,
            attack_dev=4,
            min_attack_test=4,
            min_attack_test_per_group=2,
        ),
        calibration_seeds=(2000,),
        primary_calibration_seed=2000,
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
