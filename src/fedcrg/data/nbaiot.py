"""N-BaIoT dataset adapter: nine fixed UCI device directories loaded in
preserved source-file row order.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.data.datasets import ClientData, DatasetAdapter, DatasetDiscovery, stable_row_id
from fedcrg.types import (
    AttackGroupId,
    ChronologyStatus,
    ClientId,
    DataIntegrityError,
    DatasetId,
    FailureCode,
    FeatureCount,
    PreparedColumn,
)

NBAIOT_DEVICES = {
    "nb01": ("danmini", "doorbell"),
    "nb02": ("ennio", "doorbell"),
    "nb03": ("ecobee", "thermostat"),
    "nb04": ("philips", "b120n", "baby", "monitor"),
    "nb05": ("provision", "pt737", "security", "camera"),
    "nb06": ("provision", "pt838", "security", "camera"),
    "nb07": ("simplehome", "xcs71002", "security", "camera"),
    "nb08": ("simplehome", "xcs71003", "security", "camera"),
    "nb09": ("samsung", "snh1011", "webcam"),
}


def _normalized_name(path: Path) -> str:
    return re.sub(r"[^a-z0-9]", "", str(path).lower())


class NBaiotAdapter(DatasetAdapter):
    """Load the nine named UCI devices in preserved source-file row order."""

    def __init__(self, root: Path, expected_feature_count: FeatureCount) -> None:
        super().__init__(root)
        if expected_feature_count <= 0:
            raise ValueError("expected_feature_count must be positive")
        self.expected_feature_count = expected_feature_count
        self._directories: dict[ClientId, Path] | None = None

    @property
    def dataset_id(self) -> DatasetId:
        return DatasetId.NBAIOT

    def _map_directories(self) -> dict[ClientId, Path]:
        if self._directories is not None:
            return self._directories
        directories = DatasetDiscovery.directories(self.root)
        mapping: dict[ClientId, Path] = {}
        for client_value, tokens in NBAIOT_DEVICES.items():
            client_id = str(client_value)
            matches = [
                directory
                for directory in directories
                if all(token in _normalized_name(directory) for token in tokens)
            ]
            if len(matches) != 1:
                raise DataIntegrityError(
                    f"{FailureCode.DATASET_COUNT_MISMATCH.value}: {client_id} matched "
                    f"{len(matches)} fixed device directories"
                )
            mapping[client_id] = matches[0]
        if len({str(path) for path in mapping.values()}) != len(NBAIOT_DEVICES):
            raise DataIntegrityError(
                f"{FailureCode.DATASET_COUNT_MISMATCH.value}: fixed device mapping is not one-to-one"
            )
        if len(directories) != len(NBAIOT_DEVICES):
            raise DataIntegrityError(
                f"{FailureCode.DATASET_COUNT_MISMATCH.value}: expected nine device directories, "
                f"found {len(directories)}"
            )
        self._directories = mapping
        return mapping

    def discover_clients(self) -> tuple[ClientId, ...]:
        return tuple(self._map_directories())

    def load_client(self, client_id: ClientId) -> ClientData:
        try:
            directory = self._map_directories()[client_id]
        except KeyError as exc:
            raise DataIntegrityError(f"Unknown N-BaIoT client id: {client_id}") from exc
        files = DatasetDiscovery.csv_files(directory)
        benign_files = tuple(path for path in files if "benign" in str(path).lower())
        attack_files = tuple(path for path in files if path not in benign_files)
        if not benign_files or not attack_files:
            raise DataIntegrityError(
                f"{FailureCode.DATASET_COUNT_MISMATCH.value}: {client_id} must contain benign and attack CSVs"
            )
        benign = self._load_files(benign_files, client_id, attack_group=None)
        attacks = tuple(
            self._load_files((path,), client_id, attack_group=self._attack_group(path))
            for path in attack_files
        )
        return ClientData(
            dataset=self.dataset_id,
            client_id=client_id,
            benign=benign,
            attack=pd.concat(attacks, ignore_index=True),
            chronology=ChronologyStatus.SOURCE_ORDER_ONLY,
        )

    def source_files(self) -> tuple[Path, ...]:
        return DatasetDiscovery.csv_files(self.root)

    def _load_files(
        self,
        files: tuple[Path, ...],
        client_id: ClientId,
        attack_group: AttackGroupId | None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for path in sorted(files):
            frame = pd.read_csv(str(path))
            if frame.shape[1] != self.expected_feature_count:
                raise DataIntegrityError(
                    f"{FailureCode.FEATURE_SCHEMA_MISMATCH.value}: {path} has "
                    f"{frame.shape[1]} columns, expected {self.expected_feature_count}"
                )
            try:
                numeric = frame.apply(pd.to_numeric, errors="raise")
                assert isinstance(numeric, pd.DataFrame)
            except Exception as exc:
                raise DataIntegrityError(
                    f"{FailureCode.FEATURE_SCHEMA_MISMATCH.value}: non-numeric value in {path}"
                ) from exc
            values = numeric.to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(values).all():
                raise DataIntegrityError(
                    f"{FailureCode.NONFINITE_SCORE.value}: non-finite N-BaIoT source feature in {path}"
                )
            source = str(path.relative_to(self.root))
            numeric[PreparedColumn.ROW_ID.value] = np.array(
                [
                    stable_row_id(self.dataset_id, client_id, source, index)
                    for index in range(len(numeric))
                ],
                dtype=object,
            )
            if attack_group is not None:
                numeric[PreparedColumn.ATTACK_GROUP.value] = attack_group
            numeric[PreparedColumn.SOURCE_FILE.value] = source
            numeric[PreparedColumn.SOURCE_ROW_INDEX.value] = np.arange(len(numeric), dtype=np.int64)
            frames.append(numeric)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _attack_group(path: Path) -> AttackGroupId:
        lowered = str(path).lower()
        if "mirai" in lowered:
            family = "mirai"
        elif "gafgyt" in lowered or "bashlite" in lowered:
            family = "gafgyt"
        else:
            raise DataIntegrityError(
                f"{FailureCode.DATASET_COUNT_MISMATCH.value}: cannot derive attack subtype from {path}"
            )
        subtype = path.stem.lower().replace("benign", "").replace(family, "").strip("_")
        return f"{family}_{subtype}"
