"""CIC IoT-DIAD dataset adapter: maps device MAC addresses to client ids and loads the fixed feature set."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from fedcrg.config import DatasetConfig, DiadFeatureDerivation
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


class DiadSourceColumn(StrEnum):
    """Source columns used only for client partitioning, never as model input."""

    DEVICE_MAC = "device_mac"


class DiadTopLevelCategory(StrEnum):
    """Top-level source directory name that marks benign traffic."""

    BENIGN_TRAFFIC = "benigntraffic"


class DiadAdapter(DatasetAdapter):
    """Loads per-device DIAD client data from the packet-based CSV layout."""

    def __init__(self, root: Path, dataset: DatasetConfig) -> None:
        super().__init__(root)
        if dataset.id is not DatasetId.DIAD:
            raise DataIntegrityError("The DIAD adapter requires the DIAD dataset contract")
        if dataset.diad_client_id_mac_digest_length is None:
            raise DataIntegrityError("The DIAD contract must declare its client-id derivation")
        if not dataset.feature_names:
            raise DataIntegrityError("The DIAD contract must declare its frozen feature names")
        self.dataset = dataset
        self._client_id_mac_digest_length = dataset.diad_client_id_mac_digest_length
        self._devices: dict[ClientId, tuple[str, ...]] | None = None

    @property
    def dataset_id(self) -> DatasetId:
        return DatasetId.DIAD

    @property
    def _model_columns(self) -> tuple[FeatureName, ...]:
        return (*self.dataset.feature_names, DiadSourceColumn.DEVICE_MAC)

    @staticmethod
    def _normalized_mac(value: str) -> MacAddress:
        return value.strip().lower()

    def _client_id(self, normalized_mac: MacAddress) -> ClientId:
        """Hash the MAC into an opaque client id; the MAC itself is partition metadata and never enters the feature matrix."""
        digest = hashlib.sha256(normalized_mac.encode("ascii")).hexdigest()[
            : self._client_id_mac_digest_length
        ]
        return f"diad_{digest}"

    def _map_devices(self) -> dict[ClientId, tuple[MacAddress, ...]]:
        if self._devices is not None:
            return self._devices
        macs: dict[ClientId, set[MacAddress]] = {}
        for path in DatasetDiscovery.csv_files(self.root, recursive=True):
            try:
                column = pd.read_csv(str(path), usecols=[DiadSourceColumn.DEVICE_MAC])
            except Exception as exc:
                raise DataIntegrityError(
                    f"{FailureCode.FEATURE_MISSING}: no device_mac column in "
                    f"{path.relative_to(self.root)}"
                ) from exc
            for raw in column[DiadSourceColumn.DEVICE_MAC].dropna().astype(str):
                normalized = self._normalized_mac(raw)
                if normalized:
                    macs.setdefault(self._client_id(normalized), set()).add(normalized)
        if not macs:
            raise DataIntegrityError("DIAD root contains no device identities")
        self._devices = {client_id: tuple(sorted(values)) for client_id, values in macs.items()}
        return self._devices

    def discover_clients(self) -> tuple[ClientId, ...]:
        """List the client ids derived from device MAC addresses found in the source CSVs."""
        return tuple(self._map_devices())

    def source_files(self) -> tuple[Path, ...]:
        """List all CSV files under the DIAD root."""
        return DatasetDiscovery.csv_files(self.root, recursive=True)

    def load_client(self, client_id: ClientId) -> ClientData:
        """Load and concatenate one client's benign and attack rows across all source CSVs."""
        macs = self._map_devices().get(client_id)
        if macs is None:
            raise DataIntegrityError(f"Unknown DIAD client id: {client_id}")
        benign_frames: list[pd.DataFrame] = []
        attack_frames: list[pd.DataFrame] = []
        for path in DatasetDiscovery.csv_files(self.root, recursive=True):
            category = path.relative_to(self.root).parts[0]
            try:
                frame = pd.read_csv(str(path), usecols=self._model_columns)
            except Exception as exc:
                raise DataIntegrityError(
                    f"{FailureCode.FEATURE_MISSING}: locked DIAD feature absent in "
                    f"{path.relative_to(self.root)}"
                ) from exc
            frame.columns = [str(column).strip() for column in frame.columns]
            normalized = frame[DiadSourceColumn.DEVICE_MAC].astype(str).str.strip().str.lower()
            selected = frame.loc[normalized.isin(macs)].copy()
            if selected.empty:
                continue
            # Coerce rather than raise: DIAD's packet parser can leave sparse fields, and the
            # preprocessing stage imputes them from each client's training-row medians.
            numeric = selected[list(self.dataset.feature_names)].apply(
                pd.to_numeric, errors="coerce"
            )
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            source = str(path.relative_to(self.root))
            numeric[PreparedColumn.ROW_ID] = np.array(
                [
                    stable_row_id(self.dataset_id, client_id, source, index)
                    for index in range(len(numeric))
                ],
                dtype=object,
            )
            if category.lower() == DiadTopLevelCategory.BENIGN_TRAFFIC:
                benign_frames.append(numeric)
            else:
                numeric[PreparedColumn.ATTACK_GROUP] = category.lower()
                attack_frames.append(numeric)
            numeric[PreparedColumn.SOURCE_FILE] = source
            numeric[PreparedColumn.SOURCE_ROW_INDEX] = np.arange(len(numeric), dtype=np.int64)
        if not benign_frames and not attack_frames:
            raise DataIntegrityError(f"DIAD client {client_id} has no rows in any source file")
        benign = (
            pd.concat(benign_frames, ignore_index=True)
            if benign_frames
            else pd.DataFrame(
                columns=[
                    *self.dataset.feature_names,
                    PreparedColumn.ROW_ID,
                    PreparedColumn.SOURCE_FILE,
                    PreparedColumn.SOURCE_ROW_INDEX,
                ]
            )
        )
        attack = (
            pd.concat(attack_frames, ignore_index=True)
            if attack_frames
            else pd.DataFrame(
                columns=[
                    *self.dataset.feature_names,
                    PreparedColumn.ATTACK_GROUP,
                    PreparedColumn.SOURCE_FILE,
                    PreparedColumn.SOURCE_ROW_INDEX,
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


class ClientTrainingRowHash(BaseModel):
    """A client's training-row set hash, used to verify a feature derivation is reproducible."""

    model_config = Frozen

    client_id: ClientId
    sha256: Sha256


class ClientTrainingFrame(BaseModel):
    """One client's training rows paired with its id."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    client_id: ClientId
    frame: pd.DataFrame


class NumericSafeFeatureContract(BaseModel):
    """A numeric-only feature set derived from eligible clients' training data, with its autoencoder architecture and per-client training-row hashes."""

    model_config = Frozen

    features: tuple[FeatureName, ...]
    dimension: FeatureCount
    architecture: tuple[Dimension, ...]
    training_row_hashes: tuple[ClientTrainingRowHash, ...]

    @property
    def encoder_hidden_dims(self) -> tuple[Dimension, ...]:
        """Hidden-layer widths between the input and the bottleneck."""
        return self.architecture[1:5]


def _is_forbidden_name(column: FeatureName, derivation: DiadFeatureDerivation) -> bool:
    """Excludes direct identifiers (MAC/IP/port/protocol/text fields) and prepared-column metadata from model input."""
    excluded = frozenset(derivation.excluded_feature_exact) | set(PreparedColumn)
    lowered = column.lower()
    if column in excluded or lowered in {value.lower() for value in excluded}:
        return True
    return any(marker in lowered for marker in derivation.excluded_feature_name_markers)


def derive_numeric_safe_features(
    training_frames: tuple[ClientTrainingFrame, ...],
    derivation: DiadFeatureDerivation,
) -> NumericSafeFeatureContract:
    """Intersects numeric, non-identifier columns across all eligible clients' training rows only."""
    if not training_frames:
        raise ValueError("The derived feature contract requires eligible-client training frames")
    common = set.intersection(*(set(item.frame.columns) for item in training_frames))
    first_frame = training_frames[0].frame
    selected: list[FeatureName] = []
    for column in sorted(common):
        if _is_forbidden_name(column, derivation):
            continue
        if not pd.api.types.is_numeric_dtype(first_frame[column]):
            continue
        selected.append(column)
    if not selected:
        raise ValueError("The derived feature contract is empty")
    input_dim = len(selected)
    hidden = tuple(
        max(1, int(ratio * input_dim)) for ratio in derivation.architecture_hidden_ratios
    )
    architecture = (input_dim, *hidden, input_dim)
    hashes = tuple(
        ClientTrainingRowHash(
            client_id=item.client_id,
            sha256=hash_row_ids(
                item.frame[PreparedColumn.ROW_ID].astype(str).tolist()
                if PreparedColumn.ROW_ID in item.frame.columns
                else ()
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
