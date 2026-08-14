"""Unit tests for client eligibility evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from fedcrg.config import DatasetConfig, ExpectedBenignCounts, SplitConfig
from fedcrg.data.datasets import ClientData, ClientEligibilityEvaluator
from fedcrg.data.diad import DIAD_FEATURES
from fedcrg.types import (
    ChronologyStatus,
    ClientId,
    DatasetFeatureContractId,
    DatasetId,
    EligibilityStatus,
    FailureCode,
)

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)


def _diad_config(**split_overrides: int) -> DatasetConfig:
    split = dict(
        train_benign=2,
        reference_benign=1,
        mismatch_benign=2,
        calibration_benign=2,
        benign_guard=1,
        min_benign_test=2,
        attack_dev=500,
        min_attack_test=4,
        min_attack_test_per_group=2,
    )
    split.update(split_overrides)
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
        split=SplitConfig(**split),
        calibration_seeds=(2000,),
        primary_calibration_seed=2000,
    )


def _diad_client(
    benign_rows: int = 8000,
    malicious_rows: int = 1100,
    malicious_groups: tuple[tuple[str, int], ...] = (("a", 550), ("b", 550)),
) -> ClientData:
    benign = pd.DataFrame({feature: [1.0] * benign_rows for feature in DIAD_FEATURES})
    malicious = pd.DataFrame({feature: [2.0] * malicious_rows for feature in DIAD_FEATURES})
    rows = []
    for group, count in malicious_groups:
        rows.extend([group] * count)
    malicious["attack_group"] = rows
    return ClientData(
        dataset=DatasetId.DIAD,
        client_id=_CLIENT_ID_ADAPTER.validate_python("diad_c001"),
        benign=benign,
        attack=malicious,
        chronology=ChronologyStatus.SOURCE_ORDER_ONLY,
    )


def test_diad_eligibility_accepts_healthy_client() -> None:
    record = ClientEligibilityEvaluator().evaluate(_diad_client(), _diad_config())
    assert record.status is EligibilityStatus.ELIGIBLE
    assert record.primary_code is None
    assert record.attack_development_capacity == 1096


def test_diad_eligibility_rejects_missing_feature() -> None:
    benign = pd.DataFrame({feature: [1.0] * 8000 for feature in DIAD_FEATURES[1:]})
    malicious = pd.DataFrame({feature: [2.0] * 1100 for feature in DIAD_FEATURES})
    malicious["attack_group"] = ["a"] * 550 + ["b"] * 550
    client = ClientData(
        dataset=DatasetId.DIAD,
        client_id=_CLIENT_ID_ADAPTER.validate_python("diad_c001"),
        benign=benign,
        attack=malicious,
        chronology=ChronologyStatus.SOURCE_ORDER_ONLY,
    )
    record = ClientEligibilityEvaluator().evaluate(client, _diad_config())
    assert record.status is EligibilityStatus.EXCLUDED
    assert record.primary_code is FailureCode.FEATURE_MISSING


def test_diad_eligibility_rejects_nonfinite_training_rows() -> None:
    benign = pd.DataFrame({feature: [1.0] * 8000 for feature in DIAD_FEATURES})
    benign.loc[0:1, DIAD_FEATURES[5]] = np.nan
    malicious = pd.DataFrame({feature: [2.0] * 1100 for feature in DIAD_FEATURES})
    malicious["attack_group"] = ["a"] * 550 + ["b"] * 550
    client = ClientData(
        dataset=DatasetId.DIAD,
        client_id=_CLIENT_ID_ADAPTER.validate_python("diad_c001"),
        benign=benign,
        attack=malicious,
        chronology=ChronologyStatus.SOURCE_ORDER_ONLY,
    )
    record = ClientEligibilityEvaluator().evaluate(client, _diad_config())
    assert record.status is EligibilityStatus.EXCLUDED
    assert record.primary_code is FailureCode.FINITE_RATE_FAIL


def test_diad_eligibility_rejects_insufficient_attack_development_capacity() -> None:
    record = ClientEligibilityEvaluator().evaluate(
        _diad_client(),
        _diad_config(min_attack_test_per_group=600),
    )
    assert record.status is EligibilityStatus.EXCLUDED
    assert record.primary_code is FailureCode.ATTACK_DEV_CAPACITY_LT_500
    assert record.attack_development_capacity == 0


def test_non_diad_clients_are_always_eligible() -> None:
    client = _diad_client()
    record = ClientEligibilityEvaluator().evaluate(client, _diad_config())
    assert record.status is EligibilityStatus.ELIGIBLE
