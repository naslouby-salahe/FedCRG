from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.domain.enums import DatasetId
from fedcrg.domain.identifiers import ClientId
from fedcrg.data.diad import DIAD_FEATURES, DiadAdapter
from fedcrg.data.nbaiot import _CANONICAL_DEVICES, NBaiotAdapter

_NBAIOT_CLIENT_IDS = tuple(ClientId(value) for value in _CANONICAL_DEVICES)


def test_nbaiot_adapter_maps_exact_nine_clients_and_preserves_provenance(tmp_path: Path) -> None:
    columns = [f"f{i}" for i in range(115)]
    for tokens in _CANONICAL_DEVICES.values():
        root = tmp_path / "_".join(tokens)
        (root / "gafgyt").mkdir(parents=True)
        pd.DataFrame(np.zeros((2, 115)), columns=columns).to_csv(root / "benign_traffic.csv", index=False)
        pd.DataFrame(np.ones((2, 115)), columns=columns).to_csv(root / "gafgyt" / "combo.csv", index=False)
    adapter = NBaiotAdapter(tmp_path, 115)
    assert adapter.discover_clients() == _NBAIOT_CLIENT_IDS
    client = adapter.load_client(ClientId("nb01"))
    assert client.dataset is DatasetId.NBAIOT
    assert client.benign.shape[0] == 2
    assert client.attack["attack_group"].iloc[0] == "gafgyt_combo"
    assert client.benign["row_id"].astype(str).str.len().eq(64).all()


def test_diad_adapter_uses_hashed_public_identity_and_exact_feature_allowlist(tmp_path: Path) -> None:
    root = tmp_path / "AA:BB:CC:DD:EE:FF"
    root.mkdir()
    frame = pd.DataFrame({feature: [1.0, 2.0, 3.0] for feature in DIAD_FEATURES})
    frame["label"] = ["benign", "benign", "ddos"]
    frame["attack_category"] = ["benign", "benign", "ddos"]
    frame["device_mac"] = ["AA:BB:CC:DD:EE:FF"] * 3
    frame.to_csv(root / "packets.csv", index=False)
    adapter = DiadAdapter(tmp_path)
    client_id = adapter.discover_clients()[0]
    assert client_id.value.startswith("diad_") and len(client_id.value) == 17
    client = adapter.load_client(client_id)
    assert client.dataset is DatasetId.DIAD
    assert "device_mac" not in client.benign.columns
    assert set(DIAD_FEATURES).issubset(client.benign.columns)
    assert client.attack["attack_group"].tolist() == ["ddos"]


def test_diad_adapter_partition_cache_preserves_rows_across_multiple_files_and_chunks(
    tmp_path: Path,
) -> None:
    """Regression test for the single-pass per-client partition cache: verifies that
    reading pre-filtered shards produces identical row_id/content to a direct scan,
    across multiple source files, multiple devices, and multiple internal chunks."""
    root = tmp_path / "packets"
    root.mkdir()
    macs = ["AA:AA:AA:AA:AA:01", "AA:AA:AA:AA:AA:02"]

    def _write(path: Path, macs_for_rows: list[str], labels: list[str]) -> None:
        frame = pd.DataFrame(
            {feature: list(range(len(macs_for_rows))) for feature in DIAD_FEATURES}
        )
        frame["label"] = labels
        frame["attack_category"] = labels
        frame["device_mac"] = macs_for_rows
        frame.to_csv(path, index=False)

    _write(
        root / "benign.csv",
        macs * 6,
        ["benign"] * 12,
    )
    _write(
        root / "ddos.csv",
        macs * 4,
        ["ddos"] * 8,
    )

    adapter = DiadAdapter(tmp_path)
    adapter.chunk_size = 5
    client_ids = adapter.discover_clients()
    assert len(client_ids) == 2

    for client_id in client_ids:
        client = adapter.load_client(client_id)
        assert client.benign.shape[0] == 6
        assert client.attack.shape[0] == 4
        assert client.benign["row_id"].nunique() == 6
        assert client.attack["attack_group"].eq("ddos").all()
