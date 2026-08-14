"""Integration tests for prepared-cache materialization and reuse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.config import ExperimentConfig
from fedcrg.data.datasets import ClientData, DatasetAdapter
from fedcrg.data.preparation import PrepareData
from fedcrg.evidence.store import PreparedLayout
from fedcrg.types import ClientId, DatasetId
from tests._fixtures import NBAIOT_CLIENT_IDS, nbaiot_dataset_config, primary_experiment_config

_FEATURE_COLUMNS = [f"f{i}" for i in range(1, 116)]


class FakeAdapter(DatasetAdapter):
    @property
    def dataset_id(self) -> DatasetId:
        return DatasetId.NBAIOT

    def discover_clients(self) -> tuple[ClientId, ...]:
        return NBAIOT_CLIENT_IDS

    def source_files(self) -> tuple[Path, ...]:
        return ()

    def load_client(self, client_id: ClientId) -> ClientData:
        offset = float(NBAIOT_CLIENT_IDS.index(client_id))
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
        return ClientData(
            dataset=DatasetId.NBAIOT,
            client_id=client_id,
            benign=benign,
            attack=attack,
        )


def _config(root: Path) -> ExperimentConfig:
    dataset = nbaiot_dataset_config(
        train_benign=4,
        reference_benign=2,
        mismatch_benign=2,
        calibration_benign=2,
        benign_guard=2,
        min_benign_test=4,
        attack_dev=4,
        min_attack_test=4,
        min_attack_test_per_group=2,
        expected_benign=18,
    )
    return primary_experiment_config(root).model_copy(update={"dataset": dataset})


def test_prepare_data_writes_preprocessed_roles_and_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path / "outputs")
    preparer = PrepareData()
    manifest = preparer.ensure_prepared(
        config, tmp_path / "raw", adapter_override=FakeAdapter(tmp_path / "raw")
    )
    cache = preparer.cache_root(config, manifest)
    assert (cache / PreparedLayout.manifest_filename).exists()
    assert (cache / PreparedLayout.preprocessing_filename).exists()
    assert (cache / PreparedLayout.eligibility_filename).exists()
    train = pd.read_csv(cache / "nb01" / "train.csv")
    assert train["f1"].between(0.0, 1.0).all()
    assert train["f2"].eq(0.0).all()
    assert manifest.dataset_id is DatasetId.NBAIOT


def test_prepare_data_reuses_existing_cache(tmp_path: Path) -> None:
    config = _config(tmp_path / "outputs")
    preparer = PrepareData()
    first = preparer.ensure_prepared(
        config, tmp_path / "raw", adapter_override=FakeAdapter(tmp_path / "raw")
    )
    second = preparer.ensure_prepared(
        config, tmp_path / "raw", adapter_override=FakeAdapter(tmp_path / "raw")
    )
    assert first == second
    assert preparer.cache_root(config, first) == preparer.cache_root(config, second)
