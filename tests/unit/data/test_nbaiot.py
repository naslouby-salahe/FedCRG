"""Unit tests for the N-BaIoT dataset adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from fedcrg.data.nbaiot import NBAIOT_DEVICES, NBAIOT_FEATURE_HEADERS, NBaiotAdapter
from fedcrg.types import ClientId, DataIntegrityError, DatasetId

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_NBAIOT_CLIENT_IDS = tuple(_CLIENT_ID_ADAPTER.validate_python(value) for value in NBAIOT_DEVICES)


def test_nbaiot_adapter_maps_exact_nine_clients_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    columns = list(NBAIOT_FEATURE_HEADERS)
    for tokens in NBAIOT_DEVICES.values():
        root = tmp_path / "_".join(tokens)
        (root / "gafgyt").mkdir(parents=True)
        pd.DataFrame(np.zeros((2, 115)), columns=columns).to_csv(
            root / "benign_traffic.csv", index=False
        )
        pd.DataFrame(np.ones((2, 115)), columns=columns).to_csv(
            root / "gafgyt" / "combo.csv", index=False
        )
    adapter = NBaiotAdapter(tmp_path, 115)
    assert adapter.discover_clients() == _NBAIOT_CLIENT_IDS
    client = adapter.load_client(_CLIENT_ID_ADAPTER.validate_python("nb01"))
    assert client.dataset is DatasetId.NBAIOT
    assert client.benign.shape[0] == 2
    assert client.attack["attack_group"].iloc[0] == "gafgyt_combo"
    assert client.benign["row_id"].astype(str).str.len().eq(64).all()
    assert client.benign["source_file"].nunique() == 1


def test_nbaiot_adapter_rejects_unknown_client(tmp_path: Path) -> None:
    columns = list(NBAIOT_FEATURE_HEADERS)
    for tokens in NBAIOT_DEVICES.values():
        root = tmp_path / "_".join(tokens)
        (root / "gafgyt").mkdir(parents=True)
        pd.DataFrame(np.zeros((1, 115)), columns=columns).to_csv(
            root / "benign_traffic.csv", index=False
        )
        pd.DataFrame(np.ones((1, 115)), columns=columns).to_csv(
            root / "gafgyt" / "scan.csv", index=False
        )
    adapter = NBaiotAdapter(tmp_path, 115)
    with pytest.raises(DataIntegrityError):
        adapter.load_client(_CLIENT_ID_ADAPTER.validate_python("nb99"))
