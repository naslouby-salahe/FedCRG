"""Behavior contract: prepared-cache reuse validation and rebuild triggers.

A cache whose identity matches but whose artifacts fail any validation step
must be rebuilt with a warning; a valid cache must be reused verbatim. These
tests exercise every documented invalidation trigger.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from fedcrg.config import ExperimentConfig
from fedcrg.data.datasets import ClientData, DatasetAdapter, hash_row_ids
from fedcrg.data.preparation import PrepareData
from fedcrg.evidence.models import PreparedDatasetManifest
from fedcrg.paths import PreparedDatasetLayout
from fedcrg.types import ClientId, DataIntegrityError, DatasetId
from tests._fixtures import NBAIOT_CLIENT_IDS, nbaiot_dataset_config, primary_experiment_config

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_FEATURE_COLUMNS = [f"f{i}" for i in range(1, 116)]


class FakeAdapter(DatasetAdapter):
    """Nine clients whose raw source is a single CSV per client under data_root."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        for client in NBAIOT_CLIENT_IDS:
            client_root = root / str(client)
            client_root.mkdir(parents=True, exist_ok=True)
            (client_root / "gafgyt").mkdir(parents=True, exist_ok=True)
            benign = pd.DataFrame(
                np.full((18, 115), 5.0),
                columns=["f1", "f2"] + _FEATURE_COLUMNS[2:],
            )
            benign["f1"] = np.arange(18, dtype=float)
            benign.to_csv(client_root / "benign_traffic.csv", index=False)
            attack = pd.DataFrame(
                np.full((16, 115), 5.0),
                columns=["f1", "f2"] + _FEATURE_COLUMNS[2:],
            )
            attack["f1"] = np.arange(16, dtype=float) + 20.0
            attack["attack_group"] = ["a"] * 8 + ["b"] * 8
            attack.to_csv(client_root / "gafgyt" / "combo.csv", index=False)

    @property
    def dataset_id(self) -> DatasetId:
        return DatasetId.NBAIOT

    def discover_clients(self) -> tuple[ClientId, ...]:
        return NBAIOT_CLIENT_IDS

    def source_files(self) -> tuple[Path, ...]:
        return tuple(sorted(self.root.rglob("*.csv")))

    def load_client(self, client_id: ClientId) -> ClientData:
        client_root = self.root / str(client_id)
        benign = pd.read_csv(client_root / "benign_traffic.csv")
        attack = pd.read_csv(client_root / "gafgyt" / "combo.csv")
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


PreparedState = tuple[
    ExperimentConfig, Path, PrepareData, DatasetAdapter, Path, PreparedDatasetManifest
]


@pytest.fixture()
def prepared(tmp_path: Path) -> PreparedState:
    config = _config(tmp_path / "outputs")
    raw = tmp_path / "raw"
    raw.mkdir()
    preparer = PrepareData()
    adapter = FakeAdapter(raw)
    first = preparer.ensure_prepared(config, raw, adapter_override=adapter)
    cache = preparer.cache_root(config, first)
    return config, raw, preparer, adapter, cache, first


def _rebuild_marks_new_manifest(old: PreparedDatasetManifest, new: PreparedDatasetManifest) -> bool:
    return new.created_at != old.created_at


def test_valid_cache_is_reused_verbatim(prepared: PreparedState) -> None:
    config, raw, preparer, adapter, _cache, first = prepared
    second = preparer.ensure_prepared(config, raw, adapter_override=adapter)
    assert second == first
    assert not _rebuild_marks_new_manifest(first, second)


def test_missing_manifest_triggers_rebuild(prepared: PreparedState) -> None:
    config, raw, preparer, adapter, cache, first = prepared
    (PreparedDatasetLayout(cache).manifest).unlink()
    second = preparer.ensure_prepared(config, raw, adapter_override=adapter)
    assert _rebuild_marks_new_manifest(first, second)
    assert (PreparedDatasetLayout(cache).manifest).is_file()


def test_tampered_manifest_data_spec_hash_triggers_rebuild(prepared: PreparedState) -> None:
    import json

    config, raw, preparer, adapter, cache, first = prepared
    manifest_path = PreparedDatasetLayout(cache).manifest
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["data_spec_hash"] = "f" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    second = preparer.ensure_prepared(config, raw, adapter_override=adapter)
    assert _rebuild_marks_new_manifest(first, second)


def test_missing_role_artifact_triggers_rebuild(prepared: PreparedState) -> None:
    config, raw, preparer, adapter, cache, first = prepared
    (cache / "nb01" / "train.csv").unlink()
    second = preparer.ensure_prepared(config, raw, adapter_override=adapter)
    assert _rebuild_marks_new_manifest(first, second)
    assert (cache / "nb01" / "train.csv").is_file()


def test_changed_role_artifact_triggers_rebuild(prepared: PreparedState) -> None:
    config, raw, preparer, adapter, cache, first = prepared
    artifact = cache / "nb01" / "train.csv"
    artifact.write_text("corrupted,content\n", encoding="utf-8")
    second = preparer.ensure_prepared(config, raw, adapter_override=adapter)
    assert _rebuild_marks_new_manifest(first, second)


def test_missing_calibration_assignment_triggers_rebuild(prepared: PreparedState) -> None:
    config, raw, preparer, adapter, cache, first = prepared
    import shutil

    assignment_dir = PreparedDatasetLayout(cache).seeded_splits
    shutil.rmtree(assignment_dir)
    second = preparer.ensure_prepared(config, raw, adapter_override=adapter)
    assert _rebuild_marks_new_manifest(first, second)
    assert assignment_dir.is_dir()


def test_changed_source_file_triggers_rebuild(prepared: PreparedState) -> None:
    config, raw, preparer, adapter, _cache, first = prepared
    source = raw / "nb01" / "benign_traffic.csv"
    frame = pd.read_csv(source)
    frame["f1"] = frame["f1"] + 1000.0
    frame.to_csv(source, index=False)
    second = preparer.ensure_prepared(config, raw, adapter_override=adapter)
    assert _rebuild_marks_new_manifest(first, second)


def test_missing_source_file_is_never_silently_reused(prepared: PreparedState) -> None:
    config, raw, preparer, adapter, _cache, _first = prepared
    (raw / "nb01" / "benign_traffic.csv").unlink()
    # A missing raw source invalidates the cache; with the data gone the
    # rebuild cannot succeed, so the operation must fail loudly rather than
    # silently reusing the stale cache.
    with pytest.raises((DataIntegrityError, FileNotFoundError)):
        preparer.ensure_prepared(config, raw, adapter_override=adapter)


def test_preprocessing_parameters_are_fitted_on_training_rows_only(
    prepared: PreparedState,
) -> None:
    """Fit-row hashes must equal the hash of each client's TRAIN role rows only."""
    import json

    _config, _raw, _preparer, _adapter, cache, _first = prepared
    payload = json.loads(
        (PreparedDatasetLayout(cache).preprocessing).read_text(encoding="utf-8")
    )
    for client in payload["clients"]:
        train_path = cache / client["client_id"] / "train.csv"
        train = pd.read_csv(train_path)
        expected = hash_row_ids(train["row_id"].astype(str).tolist())
        assert client["training_row_sha256"] == expected
        # A role that is NOT the training role must not be the fit source.
        reservoir = pd.read_csv(cache / client["client_id"] / "reservoir.csv")
        assert hash_row_ids(reservoir["row_id"].astype(str).tolist()) != expected


def test_adapter_dataset_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path / "outputs")
    raw = tmp_path / "raw"
    raw.mkdir()

    class WrongDatasetAdapter(FakeAdapter):
        @property
        def dataset_id(self) -> DatasetId:
            return DatasetId.DIAD

    with pytest.raises(ValueError):
        PrepareData().ensure_prepared(config, raw, adapter_override=WrongDatasetAdapter(raw))
