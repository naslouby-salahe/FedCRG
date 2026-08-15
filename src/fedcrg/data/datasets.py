"""Dataset adapter interface, deterministic id/seed hashing, and client eligibility evaluation."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from fedcrg.config import DatasetConfig
from fedcrg.types import (
    AttackGroupId,
    CalibrationSeed,
    ChronologyStatus,
    ClientId,
    DataIntegrityError,
    DatasetFeatureContractId,
    DatasetId,
    EligibilityStatus,
    FailureCode,
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
)

Frozen = ConfigDict(frozen=True)


class DatasetDiscovery:
    """Filesystem helpers for locating dataset directories and CSV files."""

    @staticmethod
    def require_root(root: Path) -> Path:
        """Resolve a dataset root path and confirm it exists as a directory."""
        resolved = root.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        return resolved

    @staticmethod
    def directories(root: Path) -> tuple[Path, ...]:
        """List a dataset root's immediate subdirectories, sorted by path."""
        resolved = DatasetDiscovery.require_root(root)
        return tuple(sorted(path for path in resolved.iterdir() if path.is_dir()))

    @staticmethod
    def csv_files(root: Path, recursive: bool = True) -> tuple[Path, ...]:
        """List CSV files under a dataset root, sorted by path."""
        resolved = DatasetDiscovery.require_root(root)
        pattern = "**/*.csv" if recursive else "*.csv"
        files = tuple(sorted(path for path in resolved.glob(pattern) if path.is_file()))
        if not files:
            raise DataIntegrityError(f"No CSV files found under {resolved}")
        return files


class ClientData(BaseModel):
    """One client's benign and attack rows, plus how its row order was established."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    dataset: DatasetId
    client_id: ClientId
    benign: pd.DataFrame
    attack: pd.DataFrame
    chronology: ChronologyStatus = ChronologyStatus.SOURCE_ORDER_ONLY


class DatasetAdapter(ABC):
    """Loads one dataset's clients from disk into a common `ClientData` shape."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @property
    @abstractmethod
    def dataset_id(self) -> DatasetId:
        raise NotImplementedError

    @abstractmethod
    def discover_clients(self) -> tuple[ClientId, ...]:
        """List the client ids this adapter can load."""
        raise NotImplementedError

    @abstractmethod
    def load_client(self, client_id: ClientId) -> ClientData:
        """Load one client's benign and attack rows."""
        raise NotImplementedError

    @abstractmethod
    def source_files(self) -> tuple[Path, ...]:
        """List every source file this adapter reads from."""
        raise NotImplementedError


def hash_row_ids(values: Iterable[RowId | str]) -> Sha256:
    """Order-independent digest, used to assert role sets are pairwise disjoint without storing raw ids."""
    normalized = sorted(str(value) for value in values)
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_seed(text: str) -> RngSeed:
    """Hash text to an unsigned 64-bit integer usable as a NumPy PCG64 seed."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & 0xFFFFFFFFFFFFFFFF


def stable_row_id(
    dataset: DatasetId, client_id: ClientId, source: str, source_index: Position
) -> RowId:
    """Derive a stable row identifier from dataset, client, source file, and row position."""
    payload = f"{dataset}{client_id}{source}{source_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def calibration_rng(
    dataset: DatasetId,
    client_id: ClientId,
    seed: CalibrationSeed,
) -> np.random.Generator:
    """Build a deterministic RNG for permuting one client's calibration reservoir.

    The seed is derived from a hash of dataset/client/seed rather than passed directly to
    PCG64, so the permutation cannot depend on loop order or Python's randomized `hash()`.
    """
    text = f"fedcrg|{dataset}|calibration|{int(seed)}|{client_id}"
    return np.random.Generator(np.random.PCG64(hash_seed(text)))


def attack_rng(
    dataset: DatasetId,
    client_id: ClientId,
    group: AttackGroupId,
    seed: ModelSeed,
) -> np.random.Generator:
    """Build a deterministic RNG for sampling one attack group's development rows."""
    namespace = "diad" if dataset is DatasetId.DIAD else dataset
    text = f"fedcrg|{namespace}|attackdev|{int(seed)}|{client_id}|{group}"
    return np.random.Generator(np.random.PCG64(hash_seed(text)))


class EligibilityRecord(BaseModel):
    """One client's eligibility outcome, with counts and any exclusion codes."""

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
    """Eligibility outcomes for every client discovered in a dataset."""

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
    """Evaluates whether a client's data satisfies dataset-specific eligibility rules.

    N-BaIoT clients are always eligible; DIAD clients must additionally pass feature,
    finite-rate, benign/malicious count, and attack-development-capacity checks.
    """

    @staticmethod
    def model_features(config: DatasetConfig) -> tuple[FeatureName, ...]:
        """Return the frozen model feature names required by a dataset config, if any."""
        if config.id is not DatasetId.DIAD:
            return ()
        if config.feature_contract in {
            DatasetFeatureContractId.DIAD_LOCKED_86,
            DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE,
        }:
            if not config.feature_names:
                raise ValueError("The DIAD feature contract is not frozen")
            return config.feature_names
        raise ValueError(f"Unsupported DIAD feature contract: {config.feature_contract}")

    def evaluate(self, data: ClientData, config: DatasetConfig) -> EligibilityRecord:
        """Evaluate one client's eligibility for the given dataset configuration."""
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
            minimum = config.diad_finite_rate_minimum
            if minimum is not None and np.any(finite_rates < minimum):
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
    def attack_development_capacity(
        data: ClientData, reserve_per_group: PositiveCount
    ) -> SampleCount:
        """Rows available for the attack-development budget after reserving `reserve_per_group` final-test rows per group."""
        if data.attack.empty or PreparedColumn.ATTACK_GROUP not in data.attack.columns:
            return 0
        counts = data.attack[PreparedColumn.ATTACK_GROUP].astype(str).value_counts()
        return int(sum(max(0, int(count) - min(reserve_per_group, int(count))) for count in counts))
