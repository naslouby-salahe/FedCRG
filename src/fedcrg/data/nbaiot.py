"""N-BaIoT dataset adapter: nine fixed UCI device directories loaded in
preserved source-file row order.
"""

from __future__ import annotations

import re
from enum import StrEnum
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


class NbaiotFeature(StrEnum): #TODO: move to config and make it a frozen dataclass so that it can be used in the config and in the adapter.
    """Locked 115-feature contract of the canonical UCI N-BaIoT schema."""

    MI_DIR_L5_WEIGHT = "mi_dir_l5_weight"
    MI_DIR_L5_MEAN = "mi_dir_l5_mean"
    MI_DIR_L5_VARIANCE = "mi_dir_l5_variance"
    MI_DIR_L3_WEIGHT = "mi_dir_l3_weight"
    MI_DIR_L3_MEAN = "mi_dir_l3_mean"
    MI_DIR_L3_VARIANCE = "mi_dir_l3_variance"
    MI_DIR_L1_WEIGHT = "mi_dir_l1_weight"
    MI_DIR_L1_MEAN = "mi_dir_l1_mean"
    MI_DIR_L1_VARIANCE = "mi_dir_l1_variance"
    MI_DIR_L0_1_WEIGHT = "mi_dir_l0_1_weight"
    MI_DIR_L0_1_MEAN = "mi_dir_l0_1_mean"
    MI_DIR_L0_1_VARIANCE = "mi_dir_l0_1_variance"
    MI_DIR_L0_01_WEIGHT = "mi_dir_l0_01_weight"
    MI_DIR_L0_01_MEAN = "mi_dir_l0_01_mean"
    MI_DIR_L0_01_VARIANCE = "mi_dir_l0_01_variance"
    H_L5_WEIGHT = "h_l5_weight"
    H_L5_MEAN = "h_l5_mean"
    H_L5_VARIANCE = "h_l5_variance"
    H_L3_WEIGHT = "h_l3_weight"
    H_L3_MEAN = "h_l3_mean"
    H_L3_VARIANCE = "h_l3_variance"
    H_L1_WEIGHT = "h_l1_weight"
    H_L1_MEAN = "h_l1_mean"
    H_L1_VARIANCE = "h_l1_variance"
    H_L0_1_WEIGHT = "h_l0_1_weight"
    H_L0_1_MEAN = "h_l0_1_mean"
    H_L0_1_VARIANCE = "h_l0_1_variance"
    H_L0_01_WEIGHT = "h_l0_01_weight"
    H_L0_01_MEAN = "h_l0_01_mean"
    H_L0_01_VARIANCE = "h_l0_01_variance"
    HH_L5_WEIGHT = "hh_l5_weight"
    HH_L5_MEAN = "hh_l5_mean"
    HH_L5_STD = "hh_l5_std"
    HH_L5_MAGNITUDE = "hh_l5_magnitude"
    HH_L5_RADIUS = "hh_l5_radius"
    HH_L5_COVARIANCE = "hh_l5_covariance"
    HH_L5_PCC = "hh_l5_pcc"
    HH_L3_WEIGHT = "hh_l3_weight"
    HH_L3_MEAN = "hh_l3_mean"
    HH_L3_STD = "hh_l3_std"
    HH_L3_MAGNITUDE = "hh_l3_magnitude"
    HH_L3_RADIUS = "hh_l3_radius"
    HH_L3_COVARIANCE = "hh_l3_covariance"
    HH_L3_PCC = "hh_l3_pcc"
    HH_L1_WEIGHT = "hh_l1_weight"
    HH_L1_MEAN = "hh_l1_mean"
    HH_L1_STD = "hh_l1_std"
    HH_L1_MAGNITUDE = "hh_l1_magnitude"
    HH_L1_RADIUS = "hh_l1_radius"
    HH_L1_COVARIANCE = "hh_l1_covariance"
    HH_L1_PCC = "hh_l1_pcc"
    HH_L0_1_WEIGHT = "hh_l0_1_weight"
    HH_L0_1_MEAN = "hh_l0_1_mean"
    HH_L0_1_STD = "hh_l0_1_std"
    HH_L0_1_MAGNITUDE = "hh_l0_1_magnitude"
    HH_L0_1_RADIUS = "hh_l0_1_radius"
    HH_L0_1_COVARIANCE = "hh_l0_1_covariance"
    HH_L0_1_PCC = "hh_l0_1_pcc"
    HH_L0_01_WEIGHT = "hh_l0_01_weight"
    HH_L0_01_MEAN = "hh_l0_01_mean"
    HH_L0_01_STD = "hh_l0_01_std"
    HH_L0_01_MAGNITUDE = "hh_l0_01_magnitude"
    HH_L0_01_RADIUS = "hh_l0_01_radius"
    HH_L0_01_COVARIANCE = "hh_l0_01_covariance"
    HH_L0_01_PCC = "hh_l0_01_pcc"
    HH_JIT_L5_WEIGHT = "hh_jit_l5_weight"
    HH_JIT_L5_MEAN = "hh_jit_l5_mean"
    HH_JIT_L5_VARIANCE = "hh_jit_l5_variance"
    HH_JIT_L3_WEIGHT = "hh_jit_l3_weight"
    HH_JIT_L3_MEAN = "hh_jit_l3_mean"
    HH_JIT_L3_VARIANCE = "hh_jit_l3_variance"
    HH_JIT_L1_WEIGHT = "hh_jit_l1_weight"
    HH_JIT_L1_MEAN = "hh_jit_l1_mean"
    HH_JIT_L1_VARIANCE = "hh_jit_l1_variance"
    HH_JIT_L0_1_WEIGHT = "hh_jit_l0_1_weight"
    HH_JIT_L0_1_MEAN = "hh_jit_l0_1_mean"
    HH_JIT_L0_1_VARIANCE = "hh_jit_l0_1_variance"
    HH_JIT_L0_01_WEIGHT = "hh_jit_l0_01_weight"
    HH_JIT_L0_01_MEAN = "hh_jit_l0_01_mean"
    HH_JIT_L0_01_VARIANCE = "hh_jit_l0_01_variance"
    HPHP_L5_WEIGHT = "hphp_l5_weight"
    HPHP_L5_MEAN = "hphp_l5_mean"
    HPHP_L5_STD = "hphp_l5_std"
    HPHP_L5_MAGNITUDE = "hphp_l5_magnitude"
    HPHP_L5_RADIUS = "hphp_l5_radius"
    HPHP_L5_COVARIANCE = "hphp_l5_covariance"
    HPHP_L5_PCC = "hphp_l5_pcc"
    HPHP_L3_WEIGHT = "hphp_l3_weight"
    HPHP_L3_MEAN = "hphp_l3_mean"
    HPHP_L3_STD = "hphp_l3_std"
    HPHP_L3_MAGNITUDE = "hphp_l3_magnitude"
    HPHP_L3_RADIUS = "hphp_l3_radius"
    HPHP_L3_COVARIANCE = "hphp_l3_covariance"
    HPHP_L3_PCC = "hphp_l3_pcc"
    HPHP_L1_WEIGHT = "hphp_l1_weight"
    HPHP_L1_MEAN = "hphp_l1_mean"
    HPHP_L1_STD = "hphp_l1_std"
    HPHP_L1_MAGNITUDE = "hphp_l1_magnitude"
    HPHP_L1_RADIUS = "hphp_l1_radius"
    HPHP_L1_COVARIANCE = "hphp_l1_covariance"
    HPHP_L1_PCC = "hphp_l1_pcc"
    HPHP_L0_1_WEIGHT = "hphp_l0_1_weight"
    HPHP_L0_1_MEAN = "hphp_l0_1_mean"
    HPHP_L0_1_STD = "hphp_l0_1_std"
    HPHP_L0_1_MAGNITUDE = "hphp_l0_1_magnitude"
    HPHP_L0_1_RADIUS = "hphp_l0_1_radius"
    HPHP_L0_1_COVARIANCE = "hphp_l0_1_covariance"
    HPHP_L0_1_PCC = "hphp_l0_1_pcc"
    HPHP_L0_01_WEIGHT = "hphp_l0_01_weight"
    HPHP_L0_01_MEAN = "hphp_l0_01_mean"
    HPHP_L0_01_STD = "hphp_l0_01_std"
    HPHP_L0_01_MAGNITUDE = "hphp_l0_01_magnitude"
    HPHP_L0_01_RADIUS = "hphp_l0_01_radius"
    HPHP_L0_01_COVARIANCE = "hphp_l0_01_covariance"
    HPHP_L0_01_PCC = "hphp_l0_01_pcc"


NBAIOT_FEATURE_HEADERS: dict[str, NbaiotFeature] = { #TODO: move to config and make it a frozen dataclass so that it can be used in the config and in the adapter.
    "MI_dir_L5_weight": NbaiotFeature.MI_DIR_L5_WEIGHT,
    "MI_dir_L5_mean": NbaiotFeature.MI_DIR_L5_MEAN,
    "MI_dir_L5_variance": NbaiotFeature.MI_DIR_L5_VARIANCE,
    "MI_dir_L3_weight": NbaiotFeature.MI_DIR_L3_WEIGHT,
    "MI_dir_L3_mean": NbaiotFeature.MI_DIR_L3_MEAN,
    "MI_dir_L3_variance": NbaiotFeature.MI_DIR_L3_VARIANCE,
    "MI_dir_L1_weight": NbaiotFeature.MI_DIR_L1_WEIGHT,
    "MI_dir_L1_mean": NbaiotFeature.MI_DIR_L1_MEAN,
    "MI_dir_L1_variance": NbaiotFeature.MI_DIR_L1_VARIANCE,
    "MI_dir_L0.1_weight": NbaiotFeature.MI_DIR_L0_1_WEIGHT,
    "MI_dir_L0.1_mean": NbaiotFeature.MI_DIR_L0_1_MEAN,
    "MI_dir_L0.1_variance": NbaiotFeature.MI_DIR_L0_1_VARIANCE,
    "MI_dir_L0.01_weight": NbaiotFeature.MI_DIR_L0_01_WEIGHT,
    "MI_dir_L0.01_mean": NbaiotFeature.MI_DIR_L0_01_MEAN,
    "MI_dir_L0.01_variance": NbaiotFeature.MI_DIR_L0_01_VARIANCE,
    "H_L5_weight": NbaiotFeature.H_L5_WEIGHT,
    "H_L5_mean": NbaiotFeature.H_L5_MEAN,
    "H_L5_variance": NbaiotFeature.H_L5_VARIANCE,
    "H_L3_weight": NbaiotFeature.H_L3_WEIGHT,
    "H_L3_mean": NbaiotFeature.H_L3_MEAN,
    "H_L3_variance": NbaiotFeature.H_L3_VARIANCE,
    "H_L1_weight": NbaiotFeature.H_L1_WEIGHT,
    "H_L1_mean": NbaiotFeature.H_L1_MEAN,
    "H_L1_variance": NbaiotFeature.H_L1_VARIANCE,
    "H_L0.1_weight": NbaiotFeature.H_L0_1_WEIGHT,
    "H_L0.1_mean": NbaiotFeature.H_L0_1_MEAN,
    "H_L0.1_variance": NbaiotFeature.H_L0_1_VARIANCE,
    "H_L0.01_weight": NbaiotFeature.H_L0_01_WEIGHT,
    "H_L0.01_mean": NbaiotFeature.H_L0_01_MEAN,
    "H_L0.01_variance": NbaiotFeature.H_L0_01_VARIANCE,
    "HH_L5_weight": NbaiotFeature.HH_L5_WEIGHT,
    "HH_L5_mean": NbaiotFeature.HH_L5_MEAN,
    "HH_L5_std": NbaiotFeature.HH_L5_STD,
    "HH_L5_magnitude": NbaiotFeature.HH_L5_MAGNITUDE,
    "HH_L5_radius": NbaiotFeature.HH_L5_RADIUS,
    "HH_L5_covariance": NbaiotFeature.HH_L5_COVARIANCE,
    "HH_L5_pcc": NbaiotFeature.HH_L5_PCC,
    "HH_L3_weight": NbaiotFeature.HH_L3_WEIGHT,
    "HH_L3_mean": NbaiotFeature.HH_L3_MEAN,
    "HH_L3_std": NbaiotFeature.HH_L3_STD,
    "HH_L3_magnitude": NbaiotFeature.HH_L3_MAGNITUDE,
    "HH_L3_radius": NbaiotFeature.HH_L3_RADIUS,
    "HH_L3_covariance": NbaiotFeature.HH_L3_COVARIANCE,
    "HH_L3_pcc": NbaiotFeature.HH_L3_PCC,
    "HH_L1_weight": NbaiotFeature.HH_L1_WEIGHT,
    "HH_L1_mean": NbaiotFeature.HH_L1_MEAN,
    "HH_L1_std": NbaiotFeature.HH_L1_STD,
    "HH_L1_magnitude": NbaiotFeature.HH_L1_MAGNITUDE,
    "HH_L1_radius": NbaiotFeature.HH_L1_RADIUS,
    "HH_L1_covariance": NbaiotFeature.HH_L1_COVARIANCE,
    "HH_L1_pcc": NbaiotFeature.HH_L1_PCC,
    "HH_L0.1_weight": NbaiotFeature.HH_L0_1_WEIGHT,
    "HH_L0.1_mean": NbaiotFeature.HH_L0_1_MEAN,
    "HH_L0.1_std": NbaiotFeature.HH_L0_1_STD,
    "HH_L0.1_magnitude": NbaiotFeature.HH_L0_1_MAGNITUDE,
    "HH_L0.1_radius": NbaiotFeature.HH_L0_1_RADIUS,
    "HH_L0.1_covariance": NbaiotFeature.HH_L0_1_COVARIANCE,
    "HH_L0.1_pcc": NbaiotFeature.HH_L0_1_PCC,
    "HH_L0.01_weight": NbaiotFeature.HH_L0_01_WEIGHT,
    "HH_L0.01_mean": NbaiotFeature.HH_L0_01_MEAN,
    "HH_L0.01_std": NbaiotFeature.HH_L0_01_STD,
    "HH_L0.01_magnitude": NbaiotFeature.HH_L0_01_MAGNITUDE,
    "HH_L0.01_radius": NbaiotFeature.HH_L0_01_RADIUS,
    "HH_L0.01_covariance": NbaiotFeature.HH_L0_01_COVARIANCE,
    "HH_L0.01_pcc": NbaiotFeature.HH_L0_01_PCC,
    "HH_jit_L5_weight": NbaiotFeature.HH_JIT_L5_WEIGHT,
    "HH_jit_L5_mean": NbaiotFeature.HH_JIT_L5_MEAN,
    "HH_jit_L5_variance": NbaiotFeature.HH_JIT_L5_VARIANCE,
    "HH_jit_L3_weight": NbaiotFeature.HH_JIT_L3_WEIGHT,
    "HH_jit_L3_mean": NbaiotFeature.HH_JIT_L3_MEAN,
    "HH_jit_L3_variance": NbaiotFeature.HH_JIT_L3_VARIANCE,
    "HH_jit_L1_weight": NbaiotFeature.HH_JIT_L1_WEIGHT,
    "HH_jit_L1_mean": NbaiotFeature.HH_JIT_L1_MEAN,
    "HH_jit_L1_variance": NbaiotFeature.HH_JIT_L1_VARIANCE,
    "HH_jit_L0.1_weight": NbaiotFeature.HH_JIT_L0_1_WEIGHT,
    "HH_jit_L0.1_mean": NbaiotFeature.HH_JIT_L0_1_MEAN,
    "HH_jit_L0.1_variance": NbaiotFeature.HH_JIT_L0_1_VARIANCE,
    "HH_jit_L0.01_weight": NbaiotFeature.HH_JIT_L0_01_WEIGHT,
    "HH_jit_L0.01_mean": NbaiotFeature.HH_JIT_L0_01_MEAN,
    "HH_jit_L0.01_variance": NbaiotFeature.HH_JIT_L0_01_VARIANCE,
    "HpHp_L5_weight": NbaiotFeature.HPHP_L5_WEIGHT,
    "HpHp_L5_mean": NbaiotFeature.HPHP_L5_MEAN,
    "HpHp_L5_std": NbaiotFeature.HPHP_L5_STD,
    "HpHp_L5_magnitude": NbaiotFeature.HPHP_L5_MAGNITUDE,
    "HpHp_L5_radius": NbaiotFeature.HPHP_L5_RADIUS,
    "HpHp_L5_covariance": NbaiotFeature.HPHP_L5_COVARIANCE,
    "HpHp_L5_pcc": NbaiotFeature.HPHP_L5_PCC,
    "HpHp_L3_weight": NbaiotFeature.HPHP_L3_WEIGHT,
    "HpHp_L3_mean": NbaiotFeature.HPHP_L3_MEAN,
    "HpHp_L3_std": NbaiotFeature.HPHP_L3_STD,
    "HpHp_L3_magnitude": NbaiotFeature.HPHP_L3_MAGNITUDE,
    "HpHp_L3_radius": NbaiotFeature.HPHP_L3_RADIUS,
    "HpHp_L3_covariance": NbaiotFeature.HPHP_L3_COVARIANCE,
    "HpHp_L3_pcc": NbaiotFeature.HPHP_L3_PCC,
    "HpHp_L1_weight": NbaiotFeature.HPHP_L1_WEIGHT,
    "HpHp_L1_mean": NbaiotFeature.HPHP_L1_MEAN,
    "HpHp_L1_std": NbaiotFeature.HPHP_L1_STD,
    "HpHp_L1_magnitude": NbaiotFeature.HPHP_L1_MAGNITUDE,
    "HpHp_L1_radius": NbaiotFeature.HPHP_L1_RADIUS,
    "HpHp_L1_covariance": NbaiotFeature.HPHP_L1_COVARIANCE,
    "HpHp_L1_pcc": NbaiotFeature.HPHP_L1_PCC,
    "HpHp_L0.1_weight": NbaiotFeature.HPHP_L0_1_WEIGHT,
    "HpHp_L0.1_mean": NbaiotFeature.HPHP_L0_1_MEAN,
    "HpHp_L0.1_std": NbaiotFeature.HPHP_L0_1_STD,
    "HpHp_L0.1_magnitude": NbaiotFeature.HPHP_L0_1_MAGNITUDE,
    "HpHp_L0.1_radius": NbaiotFeature.HPHP_L0_1_RADIUS,
    "HpHp_L0.1_covariance": NbaiotFeature.HPHP_L0_1_COVARIANCE,
    "HpHp_L0.1_pcc": NbaiotFeature.HPHP_L0_1_PCC,
    "HpHp_L0.01_weight": NbaiotFeature.HPHP_L0_01_WEIGHT,
    "HpHp_L0.01_mean": NbaiotFeature.HPHP_L0_01_MEAN,
    "HpHp_L0.01_std": NbaiotFeature.HPHP_L0_01_STD,
    "HpHp_L0.01_magnitude": NbaiotFeature.HPHP_L0_01_MAGNITUDE,
    "HpHp_L0.01_radius": NbaiotFeature.HPHP_L0_01_RADIUS,
    "HpHp_L0.01_covariance": NbaiotFeature.HPHP_L0_01_COVARIANCE,
    "HpHp_L0.01_pcc": NbaiotFeature.HPHP_L0_01_PCC,
}

NBAIOT_FEATURES = tuple(feature.value for feature in NbaiotFeature) #TODO: move to config and make it a frozen dataclass so that it can be used in the config and in the adapter.

NBAIOT_DEVICES = { #TODO: move to config and make it a frozen dataclass so that it can be used in the config and in the adapter.
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


def _normalized_name(path: Path) -> str: #TODO: don't use str primitive
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
            raw_columns = list(frame.columns)
            if raw_columns != list(NBAIOT_FEATURE_HEADERS):
                raise DataIntegrityError(
                    f"{FailureCode.FEATURE_SCHEMA_MISMATCH.value}: {path} headers do not "
                    "match the locked UCI N-BaIoT feature contract"
                )
            frame = frame.rename(
                columns={raw: feature.value for raw, feature in NBAIOT_FEATURE_HEADERS.items()}
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
        if "mirai" in lowered: #TODO: Don't use hardcoded strings. Use enum 
            family = "mirai"
        elif "gafgyt" in lowered or "bashlite" in lowered:
            family = "gafgyt" #TODO: Don't use hardcoded strings. Use enum
        else:
            raise DataIntegrityError(
                f"{FailureCode.DATASET_COUNT_MISMATCH.value}: cannot derive attack subtype from {path}"
            )
        subtype = path.stem.lower().replace("benign", "").replace(family, "").strip("_") #TODO: Don't use hardcoded strings. Use enum
        return f"{family}_{subtype}"
