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
    DEVICE_MAC = "device_mac"


class DiadTopLevelCategory(StrEnum):
    BENIGN_TRAFFIC = "benigntraffic"


class DiadAdapter(DatasetAdapter):
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
        return (*self.dataset.feature_names, DiadSourceColumn.DEVICE_MAC.value)

    @staticmethod
    def _normalized_mac(value: str) -> MacAddress:
        return value.strip().lower()

    def _client_id(self, normalized_mac: MacAddress) -> ClientId:
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
                column = pd.read_csv(str(path), usecols=[DiadSourceColumn.DEVICE_MAC.value])
            except Exception as exc:
                raise DataIntegrityError(
                    f"{FailureCode.FEATURE_MISSING.value}: no device_mac column in "
                    f"{path.relative_to(self.root)}"
                ) from exc
            for raw in column[DiadSourceColumn.DEVICE_MAC.value].dropna().astype(str):
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
                frame = pd.read_csv(str(path), usecols=self._model_columns)
            except Exception as exc:
                raise DataIntegrityError(
                    f"{FailureCode.FEATURE_MISSING.value}: locked DIAD feature absent in "
                    f"{path.relative_to(self.root)}"
                ) from exc
            frame.columns = [str(column).strip() for column in frame.columns]
            normalized = (
                frame[DiadSourceColumn.DEVICE_MAC.value].astype(str).str.strip().str.lower()
            )
            selected = frame.loc[normalized.isin(macs)].copy()
            if selected.empty:
                continue
            numeric = selected[list(self.dataset.feature_names)].apply(
                pd.to_numeric, errors="coerce"
            )
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            source = str(path.relative_to(self.root))
            numeric[PreparedColumn.ROW_ID.value] = np.array(
                [
                    stable_row_id(self.dataset_id, client_id, source, index)
                    for index in range(len(numeric))
                ],
                dtype=object,
            )
            if category.lower() == DiadTopLevelCategory.BENIGN_TRAFFIC.value:
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
                    *self.dataset.feature_names,
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
                    *self.dataset.feature_names,
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


class ClientTrainingRowHash(BaseModel):
    model_config = Frozen

    client_id: ClientId
    sha256: Sha256


class ClientTrainingFrame(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    client_id: ClientId
    frame: pd.DataFrame


class NumericSafeFeatureContract(BaseModel):
    model_config = Frozen

    features: tuple[FeatureName, ...]
    dimension: FeatureCount
    architecture: tuple[Dimension, ...]
    training_row_hashes: tuple[ClientTrainingRowHash, ...]

    @property
    def encoder_hidden_dims(self) -> tuple[Dimension, ...]:
        return self.architecture[1:5]


def _is_forbidden_name(column: FeatureName, derivation: DiadFeatureDerivation) -> bool:
    excluded = frozenset(derivation.excluded_feature_exact) | {
        item.value for item in PreparedColumn
    }
    lowered = column.lower()
    if column in excluded or lowered in {value.lower() for value in excluded}:
        return True
    return any(marker in lowered for marker in derivation.excluded_feature_name_markers)


def derive_numeric_safe_features(
    training_frames: tuple[ClientTrainingFrame, ...],
    derivation: DiadFeatureDerivation,
) -> NumericSafeFeatureContract:
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
