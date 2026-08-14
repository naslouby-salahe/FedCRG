"""Contract: row identities and source provenance are machine-independent.

The raw-data root may be reached through a symlink or differ between
checkouts; row ids must hash the source path RELATIVE to the dataset root,
never the absolute resolved path, so a clean checkout reproduces identical
prepared artifacts (roadmap 7.1.4, 14.4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.data.datasets import NBaiotAdapter
from fedcrg.types import ClientId
from pydantic import TypeAdapter


_NINE_DEVICES = (
    "Danmini_Doorbell",
    "Ennio_Doorbell",
    "Ecobee_Thermostat",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
)


def _write_device(root: Path, name: str) -> None:
    device = root / name
    (device / "gafgyt_attacks").mkdir(parents=True)
    (device / "mirai_attacks").mkdir(parents=True)
    columns = [f"f{i}" for i in range(115)]
    benign = pd.DataFrame(np.zeros((4, 115), dtype=np.float64), columns=columns)
    benign.to_csv(device / "benign_traffic.csv", index=False)
    for subtype in ("combo", "junk", "scan", "tcp", "udp"):
        frame = pd.DataFrame(np.zeros((2, 115), dtype=np.float64), columns=columns)
        frame.to_csv(device / "gafgyt_attacks" / f"{subtype}.csv", index=False)
    for subtype in ("ack", "scan", "syn", "udp", "udpplain"):
        frame = pd.DataFrame(np.zeros((2, 115), dtype=np.float64), columns=columns)
        frame.to_csv(device / "mirai_attacks" / f"{subtype}.csv", index=False)


def _write_all_devices(root: Path) -> None:
    for name in _NINE_DEVICES:
        _write_device(root, name)


def test_row_ids_are_relative_to_the_dataset_root(tmp_path: Path) -> None:
    first_root = tmp_path / "checkout_a" / "N-BaIoT"
    second_root = tmp_path / "checkout_b" / "N-BaIoT"
    for root in (first_root, second_root):
        _write_all_devices(root)

    first = NBaiotAdapter(first_root, expected_feature_count=115)
    second = NBaiotAdapter(second_root, expected_feature_count=115)
    client_id = TypeAdapter(ClientId).validate_python("nb01")

    first_data = first.load_client(client_id)
    second_data = second.load_client(client_id)

    assert first_data.benign.shape == second_data.benign.shape
    assert list(first_data.benign["row_id"]) == list(second_data.benign["row_id"])
    source = first_data.benign["source_file"].iloc[0]
    assert "checkout_a" not in source
    assert source.startswith("Danmini_Doorbell/")


def test_nbaiot_source_paths_are_relative_and_stable(tmp_path: Path) -> None:
    root = tmp_path / "data" / "raw"
    _write_all_devices(root)
    adapter = NBaiotAdapter(root, expected_feature_count=115)
    client_id = TypeAdapter(ClientId).validate_python("nb01")
    data = adapter.load_client(client_id)
    sources = set(data.benign["source_file"].astype(str)) | set(
        data.attack["source_file"].astype(str)
    )
    assert all(not source.startswith("/") for source in sources)
    assert "Danmini_Doorbell/benign_traffic.csv" in sources
    assert "Danmini_Doorbell/gafgyt_attacks/combo.csv" in sources
    assert "Danmini_Doorbell/mirai_attacks/ack.csv" in sources
