"""CIC IoT-DIAD natural-device adapter and locked 86-feature contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.core.constants import DIAD_EXPECTED_FEATURES
from fedcrg.core.enums import ChronologyStatus, DatasetId, FailureCode
from fedcrg.core.exceptions import DataIntegrityError
from fedcrg.core.ids import ClientId
from fedcrg.data.adapter import DatasetAdapter
from fedcrg.data.discovery import DatasetDiscovery
from fedcrg.data.models import ClientData
from fedcrg.data.splitting import stable_row_id

_BASE_FEATURES = (
    "inter_arrival_time",
    "time_since_previously_displayed_frame",
    "l4_tcp",
    "l4_udp",
    "ttl",
    "eth_size",
    "tcp_window_size",
    "payload_entropy",
    "payload_length",
    "l3_ip_dst_count",
    "jitter",
)
_WINDOW_TEMPLATES = (
    "stream_w_count",
    "stream_w_mean",
    "stream_w_var",
    "src_ip_w_count",
    "src_ip_w_mean",
    "src_ip_w_var",
    "src_ip_mac_w_count",
    "src_ip_mac_w_mean",
    "src_ip_mac_w_var",
    "channel_w_count",
    "channel_w_mean",
    "channel_w_var",
    "stream_jitter_w_sum",
    "stream_jitter_w_mean",
    "stream_jitter_w_var",
)
_WINDOW_FEATURES = tuple(
    name.replace("_w_", f"_{window}_")
    for window in (1, 5, 10, 30, 60)
    for name in _WINDOW_TEMPLATES
)
DIAD_FEATURES = _BASE_FEATURES + _WINDOW_FEATURES

_CAPTURE_TIME_COLUMNS = (
    "capture_time",
    "timestamp",
    "frame.time_epoch",
    "frame_time_epoch",
    "frame.time",
)
_ANOMALY_LABEL_COLUMNS = (
    "anomaly_label",
    "is_anomaly",
    "anomaly",
    "label",
    "Label",
    "attack",
)
_ATTACK_CATEGORY_COLUMNS = (
    "attack_category",
    "attack_group",
    "attack_type",
    "category",
    "Category",
    "label",
    "Label",
)
_BENIGN_LABELS = {"benign", "normal", "0", "false", "no", "none"}


class DiadAdapter(DatasetAdapter):
    """Parse natural DIAD devices without exposing identity fields to the model."""

    chunk_size = 100_000

    def __init__(self, root: Path | str) -> None:
        super().__init__(root)
        self._raw_ids_by_public_id: dict[ClientId, str] | None = None

    @property
    def dataset_id(self) -> DatasetId:
        return DatasetId.DIAD

    @property
    def model_features(self) -> tuple[str, ...]:
        return DIAD_FEATURES

    def source_files(self) -> tuple[Path, ...]:
        return DatasetDiscovery.csv_files(self.root)

    def discover_clients(self) -> tuple[ClientId, ...]:
        if self._raw_ids_by_public_id is not None:
            return tuple(sorted(self._raw_ids_by_public_id))
        raw_by_public: dict[ClientId, str] = {}
        for path in self.source_files():
            header = pd.read_csv(path, nrows=0)
            if "device_mac" not in header.columns:
                continue
            for chunk in pd.read_csv(path, usecols=["device_mac"], chunksize=self.chunk_size):
                for raw_value in chunk["device_mac"].dropna().astype(str).unique():
                    normalized = self.normalize_device_mac(raw_value)
                    public_id = self.public_client_id(normalized)
                    previous = raw_by_public.setdefault(public_id, normalized)
                    if previous != normalized:
                        raise DataIntegrityError(
                            f"{FailureCode.ID_INVALID.value}: public client-id hash collision"
                        )
        if not raw_by_public:
            raise DataIntegrityError(
                f"{FailureCode.ID_INVALID.value}: no DIAD device identities discovered"
            )
        self._raw_ids_by_public_id = raw_by_public
        return tuple(sorted(raw_by_public))

    def load_client(self, client_id: ClientId) -> ClientData:
        self.discover_clients()
        assert self._raw_ids_by_public_id is not None
        if client_id not in self._raw_ids_by_public_id:
            raise DataIntegrityError(f"{FailureCode.ID_INVALID.value}: {client_id}")
        target_mac = self._raw_ids_by_public_id[client_id]

        benign_parts: list[pd.DataFrame] = []
        attack_parts: list[pd.DataFrame] = []
        capture_available = True
        capture_parseable = True

        for path in self.source_files():
            header = pd.read_csv(path, nrows=0)
            if "device_mac" not in header.columns:
                continue
            anomaly_column = self._anomaly_label_column(header)
            category_column = self._attack_category_column(header, anomaly_column)
            capture_column = self._capture_time_column(header)
            capture_available = capture_available and capture_column is not None

            for chunk in pd.read_csv(path, chunksize=self.chunk_size):
                normalized_ids = chunk["device_mac"].astype(str).map(self.normalize_device_mac)
                selected = chunk.loc[normalized_ids == target_mac].copy()
                if selected.empty:
                    continue
                original_indices = selected.index.to_numpy(dtype=np.int64)
                anomaly_values = selected[anomaly_column]
                benign_mask = anomaly_values.map(self._is_benign).to_numpy(dtype=bool)
                model = self._model_frame(
                    selected, path, client_id, original_indices, self.model_features
                )

                if capture_column is not None:
                    parsed = pd.to_datetime(selected[capture_column], errors="coerce", utc=True)
                    capture_parseable = capture_parseable and bool(parsed.notna().all())
                    model["capture_time"] = parsed.astype("string").to_numpy()
                else:
                    capture_parseable = False

                benign_parts.append(model.loc[benign_mask].drop(columns=["attack_group"], errors="ignore"))
                malicious = model.loc[~benign_mask].copy()
                if not malicious.empty:
                    malicious["attack_group"] = (
                        selected.loc[~benign_mask, category_column]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .to_numpy()
                    )
                    if malicious["attack_group"].isin(_BENIGN_LABELS).any():
                        raise DataIntegrityError(
                            f"{FailureCode.FEATURE_SCHEMA_MISMATCH.value}: malicious rows have benign category labels"
                        )
                    attack_parts.append(malicious)

        if not benign_parts:
            raise DataIntegrityError(
                f"{FailureCode.BENIGN_COUNT_LT_7800.value}: no benign rows found for {client_id}"
            )
        benign = pd.concat(benign_parts, ignore_index=True)
        attack = pd.concat(attack_parts, ignore_index=True) if attack_parts else pd.DataFrame()

        if capture_available and capture_parseable and "capture_time" in benign.columns:
            benign = benign.sort_values(
                ["capture_time", "source_file", "source_row_index"],
                kind="stable",
            ).reset_index(drop=True)
            chronology = ChronologyStatus.VERIFIED
        else:
            benign = benign.sort_values(
                ["source_file", "source_row_index"],
                kind="stable",
            ).reset_index(drop=True)
            chronology = ChronologyStatus.SOURCE_ORDER_ONLY
        if not attack.empty:
            attack = attack.sort_values(
                ["source_file", "source_row_index"],
                kind="stable",
            ).reset_index(drop=True)
        return ClientData(
            dataset=self.dataset_id,
            client_id=client_id,
            benign=benign,
            attack=attack,
            chronology=chronology,
        )

    def load_training_schema(self, client_id: ClientId, train_count: int = 2000) -> pd.DataFrame:
        """Load only benign training-schema rows for the pre-outcome R14 feature derivation."""
        self.discover_clients()
        assert self._raw_ids_by_public_id is not None
        if client_id not in self._raw_ids_by_public_id:
            raise DataIntegrityError(f"{FailureCode.ID_INVALID.value}: {client_id}")
        target_mac = self._raw_ids_by_public_id[client_id]
        parts: list[pd.DataFrame] = []
        chronology_available = True
        chronology_parseable = True
        for path in self.source_files():
            header = pd.read_csv(path, nrows=0)
            if "device_mac" not in header.columns:
                continue
            anomaly_column = self._anomaly_label_column(header)
            capture_column = self._capture_time_column(header)
            chronology_available = chronology_available and capture_column is not None
            for chunk in pd.read_csv(path, chunksize=self.chunk_size):
                normalized_ids = chunk["device_mac"].astype(str).map(self.normalize_device_mac)
                selected = chunk.loc[normalized_ids == target_mac].copy()
                if selected.empty:
                    continue
                benign = selected.loc[selected[anomaly_column].map(self._is_benign)].copy()
                if benign.empty:
                    continue
                source = path.relative_to(self.root).as_posix()
                benign["_source_file"] = source
                benign["_source_row_index"] = benign.index.to_numpy(dtype=np.int64)
                benign["_row_id"] = [
                    stable_row_id(self.dataset_id, client_id, source, int(index))
                    for index in benign.index
                ]
                if capture_column is not None:
                    parsed = pd.to_datetime(benign[capture_column], errors="coerce", utc=True)
                    chronology_parseable = chronology_parseable and bool(parsed.notna().all())
                    benign["_capture_time"] = parsed
                else:
                    chronology_parseable = False
                parts.append(benign)
        if not parts:
            raise DataIntegrityError(f"{FailureCode.BENIGN_COUNT_LT_7800.value}: {client_id}")
        frame = pd.concat(parts, ignore_index=True)
        if chronology_available and chronology_parseable and "_capture_time" in frame.columns:
            frame = frame.sort_values(
                ["_capture_time", "_source_file", "_source_row_index"], kind="stable"
            )
        else:
            frame = frame.sort_values(["_source_file", "_source_row_index"], kind="stable")
        if len(frame) < train_count:
            raise DataIntegrityError(f"{FailureCode.BENIGN_COUNT_LT_7800.value}: {client_id}")
        return frame.iloc[:train_count].reset_index(drop=True)

    @staticmethod
    def normalize_device_mac(device_mac: str) -> str:
        normalized = device_mac.strip().lower()
        if not normalized or normalized in {"nan", "none"}:
            raise DataIntegrityError(FailureCode.ID_INVALID.value)
        return normalized

    @staticmethod
    def public_client_id(normalized_device_mac: str) -> ClientId:
        digest = hashlib.sha256(normalized_device_mac.encode("utf-8")).hexdigest()[:12]
        return ClientId(f"diad_{digest}")

    @staticmethod
    def _capture_time_column(frame: pd.DataFrame) -> str | None:
        return next((column for column in _CAPTURE_TIME_COLUMNS if column in frame.columns), None)

    @staticmethod
    def _anomaly_label_column(frame: pd.DataFrame) -> str:
        column = next((name for name in _ANOMALY_LABEL_COLUMNS if name in frame.columns), None)
        if column is None:
            raise DataIntegrityError(
                f"{FailureCode.FEATURE_MISSING.value}: anomaly label column not found"
            )
        return column

    @staticmethod
    def _attack_category_column(frame: pd.DataFrame, anomaly_column: str) -> str:
        column = next((name for name in _ATTACK_CATEGORY_COLUMNS if name in frame.columns and name != anomaly_column), None)
        if column is None:
            raise DataIntegrityError(
                f"{FailureCode.FEATURE_MISSING.value}: attack-category column not found"
            )
        return column

    @staticmethod
    def _is_benign(value: object) -> bool:
        return str(value).strip().lower() in _BENIGN_LABELS

    def _model_frame(
        self,
        frame: pd.DataFrame,
        path: Path,
        client_id: ClientId,
        source_indices: np.ndarray,
        feature_names: tuple[str, ...],
    ) -> pd.DataFrame:
        missing = [feature for feature in feature_names if feature not in frame.columns]
        if missing:
            preview = ", ".join(missing[:5])
            raise DataIntegrityError(
                f"{FailureCode.FEATURE_MISSING.value}: {len(missing)} required features absent ({preview})"
            )
        if feature_names == DIAD_FEATURES and len(DIAD_FEATURES) != DIAD_EXPECTED_FEATURES:
            raise RuntimeError("Internal DIAD feature contract is not 86 columns")
        model = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        model = pd.DataFrame(model.replace([np.inf, -np.inf], np.nan))
        source = path.relative_to(self.root).as_posix()
        model["row_id"] = np.array(
            [
                stable_row_id(self.dataset_id, client_id, source, int(index))
                for index in source_indices
            ],
            dtype=object,
        )
        model["source_file"] = source
        model["source_row_index"] = source_indices
        return model


class DiadFeatureSensitivityAdapter(DiadAdapter):
    """DIAD adapter using a frozen R14 training-schema-derived numeric feature list."""

    def __init__(self, root: Path | str, feature_names: tuple[str, ...]) -> None:
        if not feature_names:
            raise ValueError("R14 feature adapter requires at least one feature")
        super().__init__(root)
        self._feature_names = feature_names

    @property
    def model_features(self) -> tuple[str, ...]:
        return self._feature_names
