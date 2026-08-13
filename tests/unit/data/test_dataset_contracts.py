from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.core.constants import NBAIOT_CLIENT_IDS
from fedcrg.core.enums import DatasetId
from fedcrg.data.datasets.diad import DIAD_FEATURES, DiadAdapter
from fedcrg.data.datasets.nbaiot import NBaiotAdapter


def test_nbaiot_adapter_maps_exact_nine_clients_and_preserves_provenance(tmp_path: Path) -> None:
    columns = [f"f{i}" for i in range(115)]
    for client_id in NBAIOT_CLIENT_IDS:
        root = tmp_path / client_id
        (root / "gafgyt").mkdir(parents=True)
        pd.DataFrame(np.zeros((2, 115)), columns=columns).to_csv(root / "benign_traffic.csv", index=False)
        pd.DataFrame(np.ones((2, 115)), columns=columns).to_csv(root / "gafgyt" / "combo.csv", index=False)
    adapter = NBaiotAdapter(tmp_path)
    assert adapter.discover_clients() == tuple(NBAIOT_CLIENT_IDS)
    client = adapter.load_client("nb01")
    assert client.dataset is DatasetId.NBAIOT
    assert client.benign.shape[0] == 2
    assert client.attack["attack_group"].iloc[0] == "gafgyt:combo"
    assert client.benign["row_id"].str.len().eq(64).all()


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
    assert client_id.startswith("diad_") and len(client_id) == 17
    client = adapter.load_client(client_id)
    assert client.dataset is DatasetId.DIAD
    assert "device_mac" not in client.benign.columns
    assert set(DIAD_FEATURES).issubset(client.benign.columns)
    assert client.attack["attack_group"].tolist() == ["ddos"]
