"""Dataset adapters, discovery, split construction, eligibility evaluation,
and the DIAD R14 training-schema-only feature contract.

All split and calibration-role construction is deterministic: stable row ids
from dataset/client/source/index, seeded permutation for calibration roles,
and disjoint role frames validated by row-id hashing.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from abc import ABC, abstractmethod
from collections.abc import Iterable
from math import floor
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from fedcrg.config import DatasetConfig
from fedcrg.types import (
    AttackGroupId,
    CalibrationAssignmentMode,
    CalibrationReadinessState,
    CalibrationSeed,
    ChronologyStatus,
    ClientId,
    DataIntegrityError,
    DataRole,
    DatasetFeatureContractId,
    DatasetId,
    Dimension,
    EligibilityStatus,
    FailureCode,
    FeatureCount,
    FeatureName,
    ModelSeed,
    NonNegativeCount,
    PositiveCount,
    Position,
    PreparedColumn,
    RngSeed,
    RowId,
    SampleCount,
    Sha256,
    SyntheticDistribution,
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


class DatasetDiscovery:
    """Resolve dataset files without embedding workstation-specific paths."""

    @staticmethod
    def require_root(root: Path) -> Path:
        resolved = root.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        return resolved

    @staticmethod
    def directories(root: Path) -> tuple[Path, ...]:
        resolved = DatasetDiscovery.require_root(root)
        return tuple(sorted(path for path in resolved.iterdir() if path.is_dir()))

    @staticmethod
    def csv_files(root: Path, recursive: bool = True) -> tuple[Path, ...]:
        resolved = DatasetDiscovery.require_root(root)
        pattern = "**/*.csv" if recursive else "*.csv"
        files = tuple(sorted(path for path in resolved.glob(pattern) if path.is_file()))
        if not files:
            raise DataIntegrityError(f"No CSV files found under {resolved}")
        return files


class ClientData(BaseModel):
    """Benign and attack frames for one discovered client."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    dataset: DatasetId
    client_id: ClientId
    benign: pd.DataFrame
    attack: pd.DataFrame
    chronology: ChronologyStatus = ChronologyStatus.SOURCE_ORDER_ONLY


class DatasetAdapter(ABC):
    """Discover clients and load per-client source frames."""
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @property
    @abstractmethod
    def dataset_id(self) -> DatasetId:
        raise NotImplementedError

    @abstractmethod
    def discover_clients(self) -> tuple[ClientId, ...]:
        raise NotImplementedError

    @abstractmethod
    def load_client(self, client_id: ClientId) -> ClientData:
        raise NotImplementedError

    @abstractmethod
    def source_files(self) -> tuple[Path, ...]:
        raise NotImplementedError


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> Sha256:
    """SHA-256 of one file using an IO-sized read chunk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_row_ids(values: Iterable[RowId | str]) -> Sha256:
    """SHA-256 over a stable row-id sequence."""
    normalized = sorted(str(value) for value in values)
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_seed(text: str) -> RngSeed:
    """Deterministic 64-bit RNG seed from one text source."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & 0xFFFFFFFFFFFFFFFF


def stable_row_id(
    dataset: DatasetId, client_id: ClientId, source: str, source_index: Position
) -> RowId:
    """Stable hash-based row identifier for one source position."""
    payload = f"{dataset.value}{client_id}{source}{source_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calibration_rng(
    dataset: DatasetId,
    client_id: ClientId,
    seed: CalibrationSeed,
) -> np.random.Generator:
    """Deterministic per-client calibration RNG."""
    text = f"fedcrg|{dataset.value}|calibration|{int(seed)}|{client_id}"
    return np.random.Generator(np.random.PCG64(hash_seed(text)))


def attack_rng(
    dataset: DatasetId,
    client_id: ClientId,
    group: AttackGroupId,
    seed: ModelSeed,
) -> np.random.Generator:
    """Deterministic per-client attack-split RNG.

    The seed is derived per (dataset, client, attack group) so development
    sampling never depends on loop ordering or on the process-randomized
    ``hash()`` function, and different clients never share one sampling stream.
    """
    namespace = "diad" if dataset is DatasetId.DIAD else dataset.value
    text = f"fedcrg|{namespace}|attackdev|{int(seed)}|{client_id}|{group}"
    return np.random.Generator(np.random.PCG64(hash_seed(text)))


class RoleFrame(BaseModel):
    """One role-labeled frame before split assignment."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    role: DataRole
    frame: pd.DataFrame


class ClientSplits(BaseModel):
    """Immutable per-client split frames for every role."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    client_id: ClientId
    roles: tuple[RoleFrame, ...]

    def get(self, role: DataRole) -> pd.DataFrame:
        for item in self.roles:
            if item.role is role:
                return item.frame
        raise KeyError(role.value)

    def try_get(self, role: DataRole) -> pd.DataFrame | None:
        for item in self.roles:
            if item.role is role:
                return item.frame
        return None


class RolePositions(BaseModel):
    """One role's assigned row positions and their hash."""
    model_config = Frozen

    role: DataRole
    positions: tuple[Position, ...]
    row_id_hash: Sha256


class CalibrationRoleAssignment(BaseModel):
    """Seeded calibration-role assignment for one client."""
    model_config = Frozen

    client_id: ClientId
    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    roles: tuple[RolePositions, ...]

    def positions_for(self, role: DataRole) -> tuple[Position, ...]:
        if role not in {
            DataRole.REFERENCE,
            DataRole.MISMATCH,
            DataRole.CALIBRATION,
            DataRole.BENIGN_GUARD,
        }:
            raise ValueError(f"{role.value} is not a calibration-reservoir role")
        for item in self.roles:
            if item.role is role:
                return item.positions
        raise KeyError(role.value)

    def row_id_hash_for(self, role: DataRole) -> Sha256:
        for item in self.roles:
            if item.role is role:
                return item.row_id_hash
        raise KeyError(role.value)


def validate_split_disjointness(
    splits: ClientSplits,
    row_id_column: PreparedColumn = PreparedColumn.ROW_ID,
) -> None:
    """Reject any row-id overlap across split roles."""
    seen: set[str] = set()
    for role in DataRole:
        frame = splits.try_get(role)
        if frame is None:
            continue
        if row_id_column not in frame.columns:
            raise DataIntegrityError(f"{role.value} is missing {row_id_column}")
        role_ids = set(frame[row_id_column].astype(str))
        overlap = seen.intersection(role_ids)
        if overlap:
            examples = sorted(overlap)[:5]
            raise DataIntegrityError(f"Split overlap in {role.value}: {examples}")
        seen.update(role_ids)


def _ensure_row_ids(
    frame: pd.DataFrame,
    dataset: DatasetId,
    client_id: ClientId,
    source: str,
) -> pd.DataFrame:
    result = frame.copy()
    if PreparedColumn.ROW_ID.value not in result.columns:
        result[PreparedColumn.ROW_ID.value] = [
            stable_row_id(dataset, client_id, source, int(index))
            for index in range(len(result))
        ]
    return result


class CalibrationAssignmentBuilder:
    """Deterministic calibration-role positions within one client's reservoir."""

    def build(
        self,
        frame: pd.DataFrame,
        dataset: DatasetId,
        client_id: ClientId,
        config: DatasetConfig,
        calibration_seed: CalibrationSeed,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> CalibrationRoleAssignment:
        split = config.split
        reservoir_total = (
            split.reference_benign
            + split.mismatch_benign
            + split.calibration_benign
            + split.benign_guard
        )
        if len(frame) != reservoir_total:
            raise DataIntegrityError(
                f"Reservoir row count {len(frame)} != {reservoir_total} for {client_id}"
            )
        positions: tuple[Position, ...] = tuple(range(reservoir_total))
        if mode is CalibrationAssignmentMode.SEEDED_PERMUTATION:
            rng = calibration_rng(dataset, client_id, calibration_seed)
            positions = tuple(int(index) for index in rng.permutation(reservoir_total))
        boundaries = (
            split.reference_benign,
            split.reference_benign + split.mismatch_benign,
            split.reference_benign + split.mismatch_benign + split.calibration_benign,
        )
        roles: list[RolePositions] = []
        for role, (start, end) in zip(
            (
                DataRole.REFERENCE,
                DataRole.MISMATCH,
                DataRole.CALIBRATION,
                DataRole.BENIGN_GUARD,
            ),
            (
                (0, boundaries[0]),
                (boundaries[0], boundaries[1]),
                (boundaries[1], boundaries[2]),
                (boundaries[2], reservoir_total),
            ),
            strict=True,
        ):
            role_positions = positions[start:end]
            row_ids = tuple(
                frame[PreparedColumn.ROW_ID.value].astype(str).iloc[index]
                for index in role_positions
            )
            roles.append(
                RolePositions(
                    role=role,
                    positions=role_positions,
                    row_id_hash=hash_row_ids(row_ids),
                )
            )
        return CalibrationRoleAssignment(
            client_id=client_id,
            calibration_seed=calibration_seed,
            mode=mode,
            roles=tuple(roles),
        )


_NBAIOT_DEVICES = {
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


def _normalized_name(path: object) -> str:
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
        for client_value, tokens in _NBAIOT_DEVICES.items():
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
        if len({str(path) for path in mapping.values()}) != len(_NBAIOT_DEVICES):
            raise DataIntegrityError(
                f"{FailureCode.DATASET_COUNT_MISMATCH.value}: fixed device mapping is not one-to-one"
            )
        if len(directories) != len(_NBAIOT_DEVICES):
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
                numeric = cast(pd.DataFrame, frame.apply(pd.to_numeric, errors="raise"))
            except Exception as exc:
                raise DataIntegrityError(
                    f"{FailureCode.FEATURE_SCHEMA_MISMATCH.value}: non-numeric value in {path}"
                ) from exc
            values = numeric.to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(values).all():
                raise DataIntegrityError(
                    f"{FailureCode.NONFINITE_SCORE.value}: non-finite N-BaIoT source feature in {path}"
                )
            source = str(path)
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
            numeric[PreparedColumn.SOURCE_ROW_INDEX.value] = np.arange(
                len(numeric), dtype=np.int64
            )
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


class DiadAdapter(DatasetAdapter):
    """Load the DIAD NetFlow-style dataset with attack-family groups."""

    @property
    def dataset_id(self) -> DatasetId:
        return DatasetId.DIAD

    def discover_clients(self) -> tuple[ClientId, ...]:
        directories = DatasetDiscovery.directories(self.root)
        clients = tuple(
            str(path.name)
            for path in directories
            if re.match(r"^c\d+$", path.name)
        )
        if not clients:
            raise DataIntegrityError("DIAD root contains no client directories")
        return clients

    def load_client(self, client_id: ClientId) -> ClientData:
        raise NotImplementedError("DIAD client loading is implemented in the acquisition pipeline")

    def source_files(self) -> tuple[Path, ...]:
        return DatasetDiscovery.csv_files(self.root, recursive=True)


class EligibilityRecord(BaseModel):
    """Eligibility audit outcome for one client."""
    model_config = Frozen

    client_id: ClientId
    status: EligibilityStatus
    benign_count: SampleCount
    malicious_count: SampleCount
    attack_development_capacity: NonNegativeCount
    primary_code: FailureCode | None = None
    secondary_codes: tuple[FailureCode, ...] = ()
    chronology: ChronologyStatus = ChronologyStatus.SOURCE_ORDER_ONLY


class EligibilityManifest(BaseModel):
    """Frozen eligibility audit across all discovered clients."""
    model_config = Frozen

    dataset_id: DatasetId
    discovered_clients: tuple[ClientId, ...]
    eligible_clients: tuple[ClientId, ...]
    records: tuple[EligibilityRecord, ...]


_DIAD_PRECEDENCE = (
    FailureCode.ID_INVALID,
    FailureCode.FEATURE_MISSING,
    FailureCode.FINITE_RATE_FAIL,
    FailureCode.BENIGN_COUNT_LT_7800,
    FailureCode.MALICIOUS_COUNT_LT_1000,
    FailureCode.ATTACK_DEV_CAPACITY_LT_500,
)


class ClientEligibilityEvaluator:
    """Evaluate dataset-contract rules before detector or threshold outcomes exist."""

    @staticmethod
    def model_features(config: DatasetConfig) -> tuple[FeatureName, ...]:
        if config.id is not DatasetId.DIAD:
            return ()
        if config.feature_contract is DatasetFeatureContractId.DIAD_LOCKED_86:
            return DIAD_FEATURES
        if config.feature_contract is DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE:
            if not config.feature_names:
                raise ValueError("R14 DIAD feature contract is not frozen")
            return config.feature_names
        raise ValueError(f"Unsupported DIAD feature contract: {config.feature_contract.value}")

    def evaluate(self, data: ClientData, config: DatasetConfig) -> EligibilityRecord:
        benign_count = len(data.benign)
        malicious_count = len(data.attack)
        if config.id is not DatasetId.DIAD:
            return EligibilityRecord(
                client_id=data.client_id,
                status=EligibilityStatus.ELIGIBLE,
                benign_count=benign_count,
                malicious_count=malicious_count,
                attack_development_capacity=malicious_count,
                chronology=data.chronology,
            )

        model_features = self.model_features(config)
        violations: list[FailureCode] = []
        missing = [column for column in model_features if column not in data.benign.columns]
        if missing:
            violations.append(FailureCode.FEATURE_MISSING)

        train = data.benign.iloc[: config.split.train_benign]
        if len(train) >= config.split.train_benign and not missing:
            values = train.loc[:, list(model_features)].to_numpy(dtype=np.float64)
            finite_rates = np.isfinite(values).mean(axis=0)
            if np.any(finite_rates < 0.99):
                violations.append(FailureCode.FINITE_RATE_FAIL)

        if config.minimum_benign_rows is not None and benign_count < config.minimum_benign_rows:
            violations.append(FailureCode.BENIGN_COUNT_LT_7800)
        if (
            config.minimum_malicious_rows is not None
            and malicious_count < config.minimum_malicious_rows
        ):
            violations.append(FailureCode.MALICIOUS_COUNT_LT_1000)

        capacity = self.attack_development_capacity(
            data,
            config.split.min_attack_test_per_group,
        )
        if capacity < config.split.attack_dev:
            violations.append(FailureCode.ATTACK_DEV_CAPACITY_LT_500)

        ordered = tuple(code for code in _DIAD_PRECEDENCE if code in violations)
        primary = ordered[0] if ordered else None
        return EligibilityRecord(
            client_id=data.client_id,
            status=(EligibilityStatus.EXCLUDED if primary else EligibilityStatus.ELIGIBLE),
            benign_count=benign_count,
            malicious_count=malicious_count,
            attack_development_capacity=capacity,
            primary_code=primary,
            secondary_codes=ordered[1:],
            chronology=data.chronology,
        )

    @staticmethod
    def attack_development_capacity(data: ClientData, reserve_per_group: PositiveCount) -> SampleCount:
        if data.attack.empty or PreparedColumn.ATTACK_GROUP.value not in data.attack.columns:
            return 0
        counts = data.attack[PreparedColumn.ATTACK_GROUP.value].astype(str).value_counts()
        return int(
            sum(max(0, int(count) - min(reserve_per_group, int(count))) for count in counts)
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


class ClientTrainingRowHash(BaseModel):
    """Stable hash of one client's training row-id sequence."""
    model_config = Frozen

    client_id: ClientId
    sha256: Sha256


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
    training_frames: dict[ClientId, pd.DataFrame],
) -> NumericSafeFeatureContract:
    """Freeze the R14 feature list using eligible clients' training rows only."""

    if not training_frames:
        raise ValueError("R14 requires eligible-client training frames")
    common = set.intersection(*(set(frame.columns) for frame in training_frames.values()))
    selected: list[str] = []
    for column in sorted(common):
        if _is_forbidden_name(column):
            continue
        if not pd.api.types.is_numeric_dtype(training_frames[next(iter(training_frames))][column]):
            continue
        selected.append(column)
    if not selected:
        raise ValueError("R14 derived an empty feature contract")
    input_dim = len(selected)
    hidden = (
        max(1, int(0.75 * input_dim)),
        max(1, int(0.50 * input_dim)),
        max(1, input_dim // 3),
        max(1, int(0.25 * input_dim)),
    )
    architecture = (input_dim, *hidden, input_dim)
    hashes = tuple(
        ClientTrainingRowHash(
            client_id=client_id,
            sha256=hash_row_ids(
                frame[PreparedColumn.ROW_ID.value].astype(str).tolist()
                if PreparedColumn.ROW_ID.value in frame.columns
                else tuple()
            ),
        )
        for client_id, frame in sorted(training_frames.items())
    )
    return NumericSafeFeatureContract(
        features=tuple(selected),
        dimension=input_dim,
        architecture=architecture,
        training_row_hashes=hashes,
    )
