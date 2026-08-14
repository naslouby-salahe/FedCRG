"""Unit tests for the DIAD packet-based acquisition adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedcrg.data.diad import DIAD_FEATURES, DiadAdapter
from fedcrg.types import ClientId, DataIntegrityError
from pydantic import TypeAdapter

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)


def _packet_frame(
    mac: str,
    rows: int,
    offset: float = 0.0,
    header_suffix: str = "",
) -> pd.DataFrame:
    data: dict[str, object] = {
        "stream": np.arange(rows),
        "device_mac": [mac] * rows,
        **{name: np.linspace(offset, offset + 1.0, rows) for name in DIAD_FEATURES},
    }
    frame = pd.DataFrame(data)
    if header_suffix:
        frame.columns = [f"{column}{header_suffix}" for column in frame.columns]
    return frame


def _write_diad_layout(root: Path) -> None:
    (root / "BenignTraffic").mkdir(parents=True)
    (root / "Mirai" / "Mirai-syn_flood").mkdir(parents=True)
    (root / "BruteForce").mkdir(parents=True)
    _packet_frame("3C:52:82:AB:CD:01", 8, offset=0.0).to_csv(
        root / "BenignTraffic" / "BenignTraffic.csv", index=False
    )
    _packet_frame("3c:52:82:ab:cd:01", 5, offset=10.0).to_csv(
        root / "Mirai" / "Mirai-syn_flood" / "Mirai-syn_flood1.csv", index=False
    )
    _packet_frame("A1:B2:C3:D4:E5:F6", 6, offset=20.0).to_csv(
        root / "BruteForce" / "BruteForce.csv", index=False
    )
    # A second benign file for the same device (case-insensitive MAC identity).
    _packet_frame("3c:52:82:ab:cd:01", 3, offset=30.0).to_csv(
        root / "BenignTraffic" / "BenignTraffic1.csv", index=False
    )


def test_diad_adapter_discovers_normalized_macs(tmp_path: Path) -> None:
    _write_diad_layout(tmp_path)
    adapter = DiadAdapter(tmp_path, expected_feature_count=86)
    clients = adapter.discover_clients()
    assert len(clients) == 2
    assert all(client_id.startswith("diad_") for client_id in clients)
    assert all(len(client_id) == 5 + 12 for client_id in clients)


def test_diad_adapter_loads_benign_and_attack_roles(tmp_path: Path) -> None:
    _write_diad_layout(tmp_path)
    adapter = DiadAdapter(tmp_path, expected_feature_count=86)
    first = adapter.discover_clients()[0]
    data = adapter.load_client(first)
    # 8 + 3 benign rows (MAC normalized case-insensitively) and 5 attack rows.
    assert len(data.benign) == 11
    assert len(data.attack) == 5
    assert sorted(data.attack["attack_group"].unique()) == ["mirai"]
    assert all(name in data.benign.columns for name in DIAD_FEATURES)
    assert "device_mac" not in data.benign.columns
    assert data.benign["row_id"].nunique() == len(data.benign)
    assert all(str(source).startswith("BenignTraffic/") for source in data.benign["source_file"])
    assert data.chronology.value == "source_order_only"


def test_diad_adapter_rejects_unknown_client(tmp_path: Path) -> None:
    _write_diad_layout(tmp_path)
    adapter = DiadAdapter(tmp_path, expected_feature_count=86)
    with pytest.raises(DataIntegrityError, match="Unknown DIAD client"):
        adapter.load_client(_CLIENT_ID_ADAPTER.validate_python("diad_ffffffffffff"))
