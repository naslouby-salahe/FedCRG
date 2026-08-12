"""Deterministic dataset-specific role allocation.

The splitter contains only protocol-defined partition rules. Filesystem discovery,
feature parsing, preprocessing, and model training live in separate components.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from fedcrg.config.models import DatasetConfig
from fedcrg.core.enums import DataRole, DatasetId
from fedcrg.core.exceptions import DataIntegrityError
from fedcrg.data.integrity import validate_split_disjointness
from fedcrg.data.models import ClientData, ClientSplits


def _hash_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & 0xFFFFFFFFFFFFFFFF


def _calibration_rng(dataset: DatasetId, client_id: str, seed: int) -> np.random.Generator:
    text = f"fedcrg|{dataset.value}|calibration|{seed}|{client_id}"
    return np.random.Generator(np.random.PCG64(_hash_seed(text)))


def _attack_rng(dataset: DatasetId, client_id: str, group: str, seed: int) -> np.random.Generator:
    if dataset is DatasetId.DIAD:
        text = f"fedcrg|diad|attackdev|{seed}|{client_id}|{group}"
        return np.random.Generator(np.random.PCG64(_hash_seed(text)))
    return np.random.Generator(np.random.PCG64(seed))


def stable_row_id(dataset: str, client_id: str, source: str, source_index: int) -> str:
    payload = f"{dataset}|{client_id}|{source}|{source_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ensure_row_ids(frame: pd.DataFrame, dataset: str, client_id: str, source: str) -> pd.DataFrame:
    result = frame.copy()
    if "row_id" not in result.columns:
        result["row_id"] = [stable_row_id(dataset, client_id, source, int(i)) for i in range(len(result))]
    return result


def _group_counts(attack: pd.DataFrame) -> dict[str, int]:
    if "attack_group" not in attack.columns:
        raise DataIntegrityError("Attack data requires an 'attack_group' column")
    labels = attack["attack_group"].astype(str)
    return {group: int((labels == group).sum()) for group in sorted(labels.unique())}


def _nbaiot_attack_split(attack: pd.DataFrame, client_id: str, dev_count: int, min_test_count: int, min_test_per_group: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = _group_counts(attack)
    if not counts:
        raise DataIntegrityError("NBAIOT_ATTACK_BUDGET_FAIL: no attack subtype is present")
    groups = tuple(counts)
    quotient, remainder = divmod(dev_count, len(groups))
    allocation = {group: quotient + (1 if index < remainder else 0) for index, group in enumerate(groups)}
    for group in groups:
        if counts[group] - allocation[group] < min_test_per_group:
            raise DataIntegrityError(f"NBAIOT_ATTACK_BUDGET_FAIL: {group} cannot retain {min_test_per_group} final-test rows")
    labels = attack["attack_group"].astype(str)
    selected: list[int] = []
    for group in groups:
        indices = attack.index[labels == group].to_numpy()
        rng = _attack_rng(DatasetId.NBAIOT, client_id, group, seed)
        chosen = rng.choice(indices, size=allocation[group], replace=False)
        selected.extend(int(value) for value in chosen)
    dev = attack.loc[sorted(selected)].copy()
    test = attack.drop(index=selected).copy()
    if len(dev) != dev_count or len(test) < min_test_count:
        raise DataIntegrityError("NBAIOT_ATTACK_BUDGET_FAIL: final attack budget is invalid")
    return dev, test


def _diad_waterfill(counts: dict[str, int], dev_count: int, reserve: int) -> dict[str, int]:
    capacities = {group: count - min(reserve, count) for group, count in counts.items()}
    if sum(capacities.values()) < dev_count:
        raise DataIntegrityError("ATTACK_DEV_CAPACITY_LT_500")
    allocation = {group: 0 for group in counts}
    for _ in range(dev_count):
        eligible = [group for group in counts if allocation[group] < capacities[group]]
        if not eligible:
            raise DataIntegrityError("ATTACK_DEV_CAPACITY_LT_500")
        minimum = min(allocation[group] for group in eligible)
        group = min(group for group in eligible if allocation[group] == minimum)
        allocation[group] += 1
    return allocation


def _diad_attack_split(attack: pd.DataFrame, client_id: str, dev_count: int, min_test_count: int, min_test_per_group: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = _group_counts(attack)
    if len(attack) < dev_count + min_test_count:
        raise DataIntegrityError("MALICIOUS_COUNT_LT_1000")
    allocation = _diad_waterfill(counts, dev_count, min_test_per_group)
    labels = attack["attack_group"].astype(str)
    selected: list[int] = []
    for group in sorted(counts):
        indices = attack.index[labels == group].to_numpy()
        rng = _attack_rng(DatasetId.DIAD, client_id, group, seed)
        chosen = rng.choice(indices, size=allocation[group], replace=False)
        selected.extend(int(value) for value in chosen)
    dev = attack.loc[sorted(selected)].copy()
    test = attack.drop(index=selected).copy()
    if len(dev) != dev_count or len(test) < min_test_count:
        raise DataIntegrityError("DIAD final attack budget is invalid")
    final_counts = _group_counts(test)
    for group, count in counts.items():
        if final_counts.get(group, 0) < min(min_test_per_group, count):
            raise DataIntegrityError(f"DIAD final test lost required evidence for {group}")
    return dev, test


class DataSplitter:
    """Apply the frozen benign-role and malicious-development rules."""

    def split(self, data: ClientData, config: DatasetConfig, calibration_seed: int) -> ClientSplits:
        split = config.split
        benign = _ensure_row_ids(data.benign, data.dataset.value, data.client_id, "benign")
        attack = _ensure_row_ids(data.attack, data.dataset.value, data.client_id, "attack")
        required_benign = split.train_benign + split.reservoir_size + split.min_benign_test
        if len(benign) < required_benign:
            raise DataIntegrityError(f"Client {data.client_id} has {len(benign)} benign rows; {required_benign} required")
        train = benign.iloc[:split.train_benign].copy()
        reservoir_start = split.train_benign
        reservoir_end = reservoir_start + split.reservoir_size
        reservoir = benign.iloc[reservoir_start:reservoir_end].copy()
        permutation = _calibration_rng(data.dataset, data.client_id, calibration_seed).permutation(len(reservoir))
        reservoir = reservoir.iloc[permutation].copy()
        cursor = 0
        roles: dict[DataRole, pd.DataFrame] = {DataRole.TRAIN: train}
        for role, count in ((DataRole.REFERENCE, split.reference_benign), (DataRole.MISMATCH, split.mismatch_benign), (DataRole.CALIBRATION, split.calibration_benign), (DataRole.BENIGN_GUARD, split.benign_guard)):
            roles[role] = reservoir.iloc[cursor:cursor + count].copy()
            cursor += count
        roles[DataRole.BENIGN_TEST] = benign.iloc[reservoir_end:].copy()
        if data.dataset is DatasetId.NBAIOT:
            attack_dev, attack_test = _nbaiot_attack_split(attack, data.client_id, split.attack_dev, split.min_attack_test, split.min_attack_test_per_group, 9001)
        elif data.dataset is DatasetId.DIAD:
            attack_dev, attack_test = _diad_attack_split(attack, data.client_id, split.attack_dev, split.min_attack_test, split.min_attack_test_per_group, 9001)
        else:
            raise DataIntegrityError(f"No split contract for {data.dataset.value}")
        roles[DataRole.ATTACK_DEV] = attack_dev
        roles[DataRole.ATTACK_TEST] = attack_test
        for role, frame in roles.items():
            frame["role"] = role.value
        result = ClientSplits(client_id=data.client_id, roles=roles)
        validate_split_disjointness(result)
        return result
