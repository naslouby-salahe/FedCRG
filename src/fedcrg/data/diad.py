"""DIAD dataset adapter and the DIAD feature contract.

Client identity comes from the normalized ``device_mac`` column; ``device_mac``
itself never enters the model tensor. Rows are ordered by (source file, source
row index) because the packet schema carries no parseable capture-time field,
so chronology is source-order only. Attack family is the official top-level
category.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from fedcrg.data.datasets import (
    ClientData,
    DatasetAdapter,
    DatasetDiscovery,
    hash_row_ids,
    stable_row_id,
)
from fedcrg.types import (
    ChronologyStatus,
    ClientId,
    DataIntegrityError,
    DatasetId,
    Dimension,
    FailureCode,
    FeatureCount,
    FeatureName,
    MacAddress,
    PreparedColumn,
    Sha256,
)

Frozen = ConfigDict(frozen=True)


class DiadFeature(StrEnum):
    """Frozen 86-feature DIAD training-schema contract."""

    INTER_ARRIVAL_TIME = "inter_arrival_time"
    TIME_SINCE_PREVIOUSLY_DISPLAYED_FRAME = "time_since_previously_displayed_frame"
    L4_TCP = "l4_tcp"
    L4_UDP = "l4_udp"
    TTL = "ttl"
    ETH_SIZE = "eth_size"
    TCP_WINDOW_SIZE = "tcp_window_size"
    PAYLOAD_ENTROPY = "payload_entropy"
    PAYLOAD_LENGTH = "payload_length"
    L3_IP_DST_COUNT = "l3_ip_dst_count"
    JITTER = "jitter"
    STREAM_1_COUNT = "stream_1_count"
    STREAM_1_MEAN = "stream_1_mean"
    STREAM_1_VAR = "stream_1_var"
    SRC_IP_1_COUNT = "src_ip_1_count"
    SRC_IP_1_MEAN = "src_ip_1_mean"
    SRC_IP_1_VAR = "src_ip_1_var"
    SRC_IP_MAC_1_COUNT = "src_ip_mac_1_count"
    SRC_IP_MAC_1_MEAN = "src_ip_mac_1_mean"
    SRC_IP_MAC_1_VAR = "src_ip_mac_1_var"
    CHANNEL_1_COUNT = "channel_1_count"
    CHANNEL_1_MEAN = "channel_1_mean"
    CHANNEL_1_VAR = "channel_1_var"
    STREAM_JITTER_1_SUM = "stream_jitter_1_sum"
    STREAM_JITTER_1_MEAN = "stream_jitter_1_mean"
    STREAM_JITTER_1_VAR = "stream_jitter_1_var"
    STREAM_5_COUNT = "stream_5_count"
    STREAM_5_MEAN = "stream_5_mean"
    STREAM_5_VAR = "stream_5_var"
    SRC_IP_5_COUNT = "src_ip_5_count"
    SRC_IP_5_MEAN = "src_ip_5_mean"
    SRC_IP_5_VAR = "src_ip_5_var"
    SRC_IP_MAC_5_COUNT = "src_ip_mac_5_count"
    SRC_IP_MAC_5_MEAN = "src_ip_mac_5_mean"
    SRC_IP_MAC_5_VAR = "src_ip_mac_5_var"
    CHANNEL_5_COUNT = "channel_5_count"
    CHANNEL_5_MEAN = "channel_5_mean"
    CHANNEL_5_VAR = "channel_5_var"
    STREAM_JITTER_5_SUM = "stream_jitter_5_sum"
    STREAM_JITTER_5_MEAN = "stream_jitter_5_mean"
    STREAM_JITTER_5_VAR = "stream_jitter_5_var"
    STREAM_10_COUNT = "stream_10_count"
    STREAM_10_MEAN = "stream_10_mean"
    STREAM_10_VAR = "stream_10_var"
    SRC_IP_10_COUNT = "src_ip_10_count"
    SRC_IP_10_MEAN = "src_ip_10_mean"
    SRC_IP_10_VAR = "src_ip_10_var"
    SRC_IP_MAC_10_COUNT = "src_ip_mac_10_count"
    SRC_IP_MAC_10_MEAN = "src_ip_mac_10_mean"
    SRC_IP_MAC_10_VAR = "src_ip_mac_10_var"
    CHANNEL_10_COUNT = "channel_10_count"
    CHANNEL_10_MEAN = "channel_10_mean"
    CHANNEL_10_VAR = "channel_10_var"
    STREAM_JITTER_10_SUM = "stream_jitter_10_sum"
    STREAM_JITTER_10_MEAN = "stream_jitter_10_mean"
    STREAM_JITTER_10_VAR = "stream_jitter_10_var"
    STREAM_30_COUNT = "stream_30_count"
    STREAM_30_MEAN = "stream_30_mean"
    STREAM_30_VAR = "stream_30_var"
    SRC_IP_30_COUNT = "src_ip_30_count"
    SRC_IP_30_MEAN = "src_ip_30_mean"
    SRC_IP_30_VAR = "src_ip_30_var"
    SRC_IP_MAC_30_COUNT = "src_ip_mac_30_count"
    SRC_IP_MAC_30_MEAN = "src_ip_mac_30_mean"
    SRC_IP_MAC_30_VAR = "src_ip_mac_30_var"
    CHANNEL_30_COUNT = "channel_30_count"
    CHANNEL_30_MEAN = "channel_30_mean"
    CHANNEL_30_VAR = "channel_30_var"
    STREAM_JITTER_30_SUM = "stream_jitter_30_sum"
    STREAM_JITTER_30_MEAN = "stream_jitter_30_mean"
    STREAM_JITTER_30_VAR = "stream_jitter_30_var"
    STREAM_60_COUNT = "stream_60_count"
    STREAM_60_MEAN = "stream_60_mean"
    STREAM_60_VAR = "stream_60_var"
    SRC_IP_60_COUNT = "src_ip_60_count"
    SRC_IP_60_MEAN = "src_ip_60_mean"
    SRC_IP_60_VAR = "src_ip_60_var"
    SRC_IP_MAC_60_COUNT = "src_ip_mac_60_count"
    SRC_IP_MAC_60_MEAN = "src_ip_mac_60_mean"
    SRC_IP_MAC_60_VAR = "src_ip_mac_60_var"
    CHANNEL_60_COUNT = "channel_60_count"
    CHANNEL_60_MEAN = "channel_60_mean"
    CHANNEL_60_VAR = "channel_60_var"
    STREAM_JITTER_60_SUM = "stream_jitter_60_sum"
    STREAM_JITTER_60_MEAN = "stream_jitter_60_mean"
    STREAM_JITTER_60_VAR = "stream_jitter_60_var"


DIAD_FEATURES = tuple(feature.value for feature in list(DiadFeature))
_CLIENT_ID_MAC_DIGEST_LENGTH = 12


class DiadAdapter(DatasetAdapter):
    """Load the CIC IoT-DIAD 2024 packet-based release into per-device clients."""

    _MODEL_COLUMNS = (*DIAD_FEATURES, "device_mac")

    def __init__(self, root: Path, expected_feature_count: FeatureCount) -> None:
        super().__init__(root)
        if expected_feature_count != len(DIAD_FEATURES):
            raise DataIntegrityError(
                f"{FailureCode.FEATURE_SCHEMA_MISMATCH.value}: DIAD expects "
                f"{len(DIAD_FEATURES)} features, received {expected_feature_count}"
            )
        self.expected_feature_count = expected_feature_count
        self._devices: dict[ClientId, tuple[str, ...]] | None = None

    @property
    def dataset_id(self) -> DatasetId:
        return DatasetId.DIAD

    @staticmethod
    def _normalized_mac(value: str) -> MacAddress:
        return value.strip().lower()

    @classmethod
    def _client_id(cls, normalized_mac: MacAddress) -> ClientId:
        digest = hashlib.sha256(normalized_mac.encode("ascii")).hexdigest()[
            :_CLIENT_ID_MAC_DIGEST_LENGTH
        ]
        return f"diad_{digest}"

    def _map_devices(self) -> dict[ClientId, tuple[MacAddress, ...]]:
        """Scan every source CSV for distinct normalized device MACs."""
        if self._devices is not None:
            return self._devices
        macs: dict[ClientId, set[MacAddress]] = {}
        for path in DatasetDiscovery.csv_files(self.root, recursive=True):
            try:
                column = pd.read_csv(str(path), usecols=["device_mac"])
            except Exception as exc:
                raise DataIntegrityError(
                    f"{FailureCode.FEATURE_MISSING.value}: no device_mac column in "
                    f"{path.relative_to(self.root)}"
                ) from exc
            for raw in column["device_mac"].dropna().astype(str):
                normalized = self._normalized_mac(raw)
                if normalized:
                    macs.setdefault(self._client_id(normalized), set()).add(normalized)
        if not macs:
            raise DataIntegrityError("DIAD root contains no device identities")
        self._devices = {client_id: tuple(sorted(values)) for client_id, values in macs.items()}
        return self._devices

    def discover_clients(self) -> tuple[ClientId, ...]:
        return tuple(self._map_devices())

    def source_files(self) -> tuple[Path, ...]:
        return DatasetDiscovery.csv_files(self.root, recursive=True)

    def load_client(self, client_id: ClientId) -> ClientData:
        macs = self._map_devices().get(client_id)
        if macs is None:
            raise DataIntegrityError(f"Unknown DIAD client id: {client_id}")
        benign_frames: list[pd.DataFrame] = []
        attack_frames: list[pd.DataFrame] = []
        for path in DatasetDiscovery.csv_files(self.root, recursive=True):
            category = path.relative_to(self.root).parts[0]
            try:
                frame = pd.read_csv(str(path), usecols=self._MODEL_COLUMNS)
            except Exception as exc:
                raise DataIntegrityError(
                    f"{FailureCode.FEATURE_MISSING.value}: locked DIAD feature absent in "
                    f"{path.relative_to(self.root)}"
                ) from exc
            frame.columns = [str(column).strip() for column in frame.columns]
            normalized = frame["device_mac"].astype(str).str.strip().str.lower()
            selected = frame.loc[normalized.isin(macs)].copy()
            if selected.empty:
                continue
            numeric = selected[list(DIAD_FEATURES)].apply(pd.to_numeric, errors="coerce")
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            source = str(path.relative_to(self.root))
            numeric[PreparedColumn.ROW_ID.value] = np.array(
                [
                    stable_row_id(self.dataset_id, client_id, source, index)
                    for index in range(len(numeric))
                ],
                dtype=object,
            )
            if category.lower() == "benigntraffic":
                benign_frames.append(numeric)
            else:
                numeric[PreparedColumn.ATTACK_GROUP.value] = category.lower()
                attack_frames.append(numeric)
            numeric[PreparedColumn.SOURCE_FILE.value] = source
            numeric[PreparedColumn.SOURCE_ROW_INDEX.value] = np.arange(len(numeric), dtype=np.int64)
        if not benign_frames and not attack_frames:
            raise DataIntegrityError(f"DIAD client {client_id} has no rows in any source file")
        benign = (
            pd.concat(benign_frames, ignore_index=True)
            if benign_frames
            else pd.DataFrame(
                columns=[
                    *DIAD_FEATURES,
                    PreparedColumn.ROW_ID.value,
                    PreparedColumn.SOURCE_FILE.value,
                    PreparedColumn.SOURCE_ROW_INDEX.value,
                ]
            )
        )
        attack = (
            pd.concat(attack_frames, ignore_index=True)
            if attack_frames
            else pd.DataFrame(
                columns=[
                    *DIAD_FEATURES,
                    PreparedColumn.ROW_ID.value,
                    PreparedColumn.ATTACK_GROUP.value,
                    PreparedColumn.SOURCE_FILE.value,
                    PreparedColumn.SOURCE_ROW_INDEX.value,
                ]
            )
        )
        return ClientData(
            dataset=self.dataset_id,
            client_id=client_id,
            benign=benign,
            attack=attack,
            chronology=ChronologyStatus.SOURCE_ORDER_ONLY,
        )


_EXCLUDED_EXACT = {
    "stream",
    "device_mac",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "port_class_dst",
    "Label",
    "anomaly",
    "is_anomaly",
    "attack",
    "verified_chronology",
    *(column.value for column in PreparedColumn),
}
_EXCLUDED_NAME_MARKERS = (
    "user_agent",
    "hostname",
    "domain_name",
    "uri",
    "mac_address",
    "ip_address",
)

_ARCHITECTURE_INPUT_RATIO_LARGE = 0.75
_ARCHITECTURE_INPUT_RATIO_MEDIUM = 0.50
_ARCHITECTURE_INPUT_RATIO_SMALL = 1.0 / 3.0
_ARCHITECTURE_INPUT_RATIO_BOTTLENECK = 0.25


class ClientTrainingRowHash(BaseModel):
    """Stable hash of one client's training row-id sequence."""

    model_config = Frozen

    client_id: ClientId
    sha256: Sha256


class ClientTrainingFrame(BaseModel):
    """One eligible client's training-role frame, keyed by client identity."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    client_id: ClientId
    frame: pd.DataFrame


class NumericSafeFeatureContract(BaseModel):
    """Frozen training-schema-only feature derivation result."""

    model_config = Frozen

    features: tuple[FeatureName, ...]
    dimension: FeatureCount
    architecture: tuple[Dimension, ...]
    training_row_hashes: tuple[ClientTrainingRowHash, ...]

    @property
    def encoder_hidden_dims(self) -> tuple[Dimension, ...]:
        return self.architecture[1:5]


def _is_forbidden_name(column: FeatureName) -> bool:
    lowered = column.lower()
    if column in _EXCLUDED_EXACT or lowered in {value.lower() for value in _EXCLUDED_EXACT}:
        return True
    return any(marker in lowered for marker in _EXCLUDED_NAME_MARKERS)


def derive_numeric_safe_features(
    training_frames: tuple[ClientTrainingFrame, ...],
) -> NumericSafeFeatureContract:
    """Freeze the training-schema-derived feature list from eligible training rows."""

    if not training_frames:
        raise ValueError("The derived feature contract requires eligible-client training frames")
    common = set.intersection(*(set(item.frame.columns) for item in training_frames))
    first_frame = training_frames[0].frame
    selected: list[FeatureName] = []
    for column in sorted(common):
        if _is_forbidden_name(column):
            continue
        if not pd.api.types.is_numeric_dtype(first_frame[column]):
            continue
        selected.append(column)
    if not selected:
        raise ValueError("The derived feature contract is empty")
    input_dim = len(selected)
    hidden = (
        max(1, int(_ARCHITECTURE_INPUT_RATIO_LARGE * input_dim)),
        max(1, int(_ARCHITECTURE_INPUT_RATIO_MEDIUM * input_dim)),
        max(1, int(_ARCHITECTURE_INPUT_RATIO_SMALL * input_dim)),
        max(1, int(_ARCHITECTURE_INPUT_RATIO_BOTTLENECK * input_dim)),
    )
    architecture = (input_dim, *hidden, input_dim)
    hashes = tuple(
        ClientTrainingRowHash(
            client_id=item.client_id,
            sha256=hash_row_ids(
                item.frame[PreparedColumn.ROW_ID.value].astype(str).tolist()
                if PreparedColumn.ROW_ID.value in item.frame.columns
                else tuple()
            ),
        )
        for item in sorted(training_frames, key=lambda entry: entry.client_id)
    )
    return NumericSafeFeatureContract(
        features=tuple(selected),
        dimension=input_dim,
        architecture=architecture,
        training_row_hashes=hashes,
    )
