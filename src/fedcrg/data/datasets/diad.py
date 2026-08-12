"""CIC IoT-DIAD natural-device adapter with a locked numeric feature contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.core.constants import DIAD_EXPECTED_FEATURES
from fedcrg.core.enums import DatasetId
from fedcrg.core.exceptions import DataIntegrityError
from fedcrg.data.adapter import DatasetAdapter
from fedcrg.data.discovery import DatasetDiscovery
from fedcrg.data.models import ClientData
from fedcrg.data.splitting import stable_row_id

_BASE_FEATURES = (
    "inter_arrival_time", "time_since_previously_displayed_frame", "l4_tcp", "l4_udp",
    "ttl", "eth_size", "tcp_window_size", "payload_entropy", "payload_length",
    "l3_ip_dst_count", "jitter",
)
_WINDOW_FEATURES = tuple(
    name.replace("_w_", f"_{window}_")
    for window in (1, 5, 10, 30, 60)
    for name in (
        "stream_w_count", "stream_w_mean", "stream_w_var", "src_ip_w_count", "src_ip_w_mean",
        "src_ip_w_var", "src_ip_mac_w_count", "src_ip_mac_w_mean", "src_ip_mac_w_var",
        "channel_w_count", "channel_w_mean", "channel_w_var", "stream_jitter_w_sum",
        "stream_jitter_w_mean", "stream_jitter_w_var",
    )
)
DIAD_FEATURES = _BASE_FEATURES + _WINDOW_FEATURES


class DiadAdapter(DatasetAdapter):
    """Load eligible DIAD clients while keeping identity fields outside model tensors."""

    @property
    def dataset_id(self) -> DatasetId:
        return DatasetId.DIAD

    def discover_clients(self) -> tuple[str, ...]:
        files = DatasetDiscovery.csv_files(self.root)
        identities: set[str] = set()
        for path in files:
            header = pd.read_csv(path, nrows=0)
            if "device_mac" not in header.columns:
                continue
            for value in pd.read_csv(path, usecols=["device_mac"])["device_mac"].dropna().astype(str):
                identities.add(self.public_client_id(value))
        if not identities:
            raise DataIntegrityError("ID_INVALID: no DIAD device identities discovered")
        return tuple(sorted(identities))

    def load_client(self, client_id: str) -> ClientData:
        benign_parts: list[pd.DataFrame] = []
        attack_parts: list[pd.DataFrame] = []
        for path in DatasetDiscovery.csv_files(self.root):
            frame = pd.read_csv(path)
            if "device_mac" not in frame.columns:
                continue
            public_ids = frame["device_mac"].astype(str).map(self.public_client_id)
            selected = frame.loc[public_ids == client_id].copy()
            if selected.empty:
                continue
            selected = self._normalize(selected, path, client_id)
            label_column = self._label_column(frame)
            labels = frame.loc[public_ids == client_id, label_column].astype(str).str.lower()
            benign_mask = labels.isin({"benign", "normal", "0", "false"})
            benign_parts.append(selected.loc[benign_mask].drop(columns=["attack_group"], errors="ignore"))
            attack_parts.append(selected.loc[~benign_mask])
        if not benign_parts:
            raise DataIntegrityError(f"BENIGN_COUNT_LT_7800: no benign rows found for {client_id}")
        benign = pd.concat(benign_parts, ignore_index=True)
        attack = pd.concat(attack_parts, ignore_index=True) if attack_parts else pd.DataFrame()
        return ClientData(dataset=self.dataset_id, client_id=client_id, benign=benign, attack=attack)

    @staticmethod
    def public_client_id(device_mac: str) -> str:
        normalized = device_mac.strip().lower()
        if not normalized:
            raise DataIntegrityError("ID_INVALID")
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
        return f"diad_{digest}"

    @staticmethod
    def _label_column(frame: pd.DataFrame) -> str:
        for candidate in ("label", "Label", "anomaly", "is_anomaly", "attack"):
            if candidate in frame.columns:
                return candidate
        raise DataIntegrityError("FEATURE_MISSING: DIAD anomaly label column not found")

    def _normalize(self, frame: pd.DataFrame, path: Path, client_id: str) -> pd.DataFrame:
        missing = [feature for feature in DIAD_FEATURES if feature not in frame.columns]
        if missing:
            raise DataIntegrityError(f"FEATURE_MISSING: {len(missing)} required DIAD features are absent")
        if len(DIAD_FEATURES) != DIAD_EXPECTED_FEATURES:
            raise RuntimeError("Internal DIAD feature contract is not 86 columns")
        model = frame.loc[:, DIAD_FEATURES].apply(pd.to_numeric, errors="coerce")
        model = model.replace([np.inf, -np.inf], np.nan)
        source = path.relative_to(self.root).as_posix()
        model["row_id"] = [stable_row_id(self.dataset_id.value, client_id, source, int(index)) for index in frame.index]
        model["source_file"] = source
        model["source_row_index"] = frame.index.to_numpy(dtype=np.int64)
        label = self._label_column(frame)
        model["attack_group"] = frame[label].astype(str).str.lower().to_numpy()
        return model
