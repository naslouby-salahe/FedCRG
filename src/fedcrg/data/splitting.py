"""Deterministic base partitioning and calibration-reservoir assignments."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from fedcrg.config.models import DatasetConfig
from fedcrg.core.constants import ATTACK_DEVELOPMENT_SEED
from fedcrg.core.enums import CalibrationAssignmentMode, DataRole, DatasetId, FailureCode
from fedcrg.core.exceptions import DataIntegrityError
from fedcrg.core.ids import ClientId, Sha256
from fedcrg.data.integrity import validate_split_disjointness
from fedcrg.data.manifests import hash_row_ids
from fedcrg.data.models import CalibrationRoleAssignment, ClientData, ClientSplits

_CALIBRATION_ROLES = (
    DataRole.REFERENCE,
    DataRole.MISMATCH,
    DataRole.CALIBRATION,
    DataRole.BENIGN_GUARD,
)


def hash_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & 0xFFFFFFFFFFFFFFFF


def calibration_rng(dataset: DatasetId, client_id: ClientId, seed: int) -> np.random.Generator:
    text = f"fedcrg|{dataset.value}|calibration|{seed}|{client_id.value}"
    return np.random.Generator(np.random.PCG64(hash_seed(text)))


def attack_rng(
    dataset: DatasetId,
    client_id: ClientId,
    group: str,
    seed: int,
) -> np.random.Generator:
    if dataset is DatasetId.DIAD:
        text = f"fedcrg|diad|attackdev|{seed}|{client_id.value}|{group}"
        return np.random.Generator(np.random.PCG64(hash_seed(text)))
    return np.random.Generator(np.random.PCG64(seed))


def stable_row_id(
    dataset: DatasetId,
    client_id: ClientId,
    source: str,
    source_index: int,
) -> str:
    payload = f"{dataset.value}{client_id.value}{source}{source_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ensure_row_ids(
    frame: pd.DataFrame,
    dataset: DatasetId,
    client_id: ClientId,
    source: str,
) -> pd.DataFrame:
    result = frame.copy()
    if "row_id" not in result.columns:
        result["row_id"] = [
            stable_row_id(dataset, client_id, source, int(index))
            for index in range(len(result))
        ]
    return result


def attack_group_counts(attack: pd.DataFrame) -> dict[str, int]:
    if "attack_group" not in attack.columns:
        raise DataIntegrityError("Attack data requires an attack_group column")
    labels = attack["attack_group"].astype(str)
    return {group: int((labels == group).sum()) for group in sorted(labels.unique())}


def allocate_nbaiot_attack_development(
    counts: dict[str, int],
    development_count: int,
    minimum_test_per_group: int,
) -> dict[str, int]:
    if not counts:
        raise DataIntegrityError(
            f"{FailureCode.NBAIOT_ATTACK_BUDGET_FAIL.value}: no attack subtype is present"
        )
    groups = tuple(sorted(counts))
    quotient, remainder = divmod(development_count, len(groups))
    allocation = {
        group: quotient + (1 if index < remainder else 0)
        for index, group in enumerate(groups)
    }
    for group, allocated in allocation.items():
        if counts[group] - allocated < minimum_test_per_group:
            raise DataIntegrityError(
                f"{FailureCode.NBAIOT_ATTACK_BUDGET_FAIL.value}: {group} cannot retain "
                f"{minimum_test_per_group} final-test rows"
            )
    if sum(allocation.values()) != development_count:
        raise RuntimeError("N-BaIoT development allocation does not sum to its locked budget")
    return allocation


def allocate_diad_attack_development(
    counts: dict[str, int],
    development_count: int,
    reserve_per_group: int,
) -> dict[str, int]:
    capacities = {
        group: count - min(reserve_per_group, count)
        for group, count in counts.items()
    }
    if sum(capacities.values()) < development_count:
        raise DataIntegrityError(FailureCode.ATTACK_DEV_CAPACITY_LT_500.value)

    allocation = {group: 0 for group in sorted(counts)}
    for _ in range(development_count):
        eligible = [
            group
            for group in allocation
            if allocation[group] < capacities[group]
        ]
        if not eligible:
            raise DataIntegrityError(FailureCode.ATTACK_DEV_CAPACITY_LT_500.value)
        minimum = min(allocation[group] for group in eligible)
        chosen = min(group for group in eligible if allocation[group] == minimum)
        allocation[chosen] += 1

    if sum(allocation.values()) != development_count:
        raise RuntimeError("DIAD water-filling allocation does not sum to the development budget")
    if any(allocation[group] > capacities[group] for group in allocation):
        raise RuntimeError("DIAD water-filling allocation exceeded a category capacity")
    return allocation


def _sample_attack_development(
    attack: pd.DataFrame,
    dataset: DatasetId,
    client_id: ClientId,
    allocation: dict[str, int],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = attack["attack_group"].astype(str)
    selected: list[int] = []
    for group in sorted(allocation):
        count = allocation[group]
        if count == 0:
            continue
        indices = attack.index[labels == group].to_numpy()
        chosen = attack_rng(dataset, client_id, group, seed).choice(
            indices,
            size=count,
            replace=False,
        )
        selected.extend(int(value) for value in chosen)
    development = attack.loc[sorted(selected)].copy()
    test = attack.drop(index=selected).copy()
    return development, test


class CalibrationAssignmentBuilder:
    """Create deterministic R/G/C/guard positions without rescoring reservoir rows."""

    def build(
        self,
        reservoir: pd.DataFrame,
        dataset: DatasetId,
        client_id: ClientId,
        config: DatasetConfig,
        calibration_seed: int,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> CalibrationRoleAssignment:
        expected = config.split.reservoir_size
        if len(reservoir) != expected:
            raise DataIntegrityError(
                f"Reservoir has {len(reservoir)} rows; expected {expected}"
            )
        if calibration_seed not in config.calibration_seeds:
            raise ValueError(f"Calibration seed {calibration_seed} is not configured")
        if mode is CalibrationAssignmentMode.SEEDED_PERMUTATION:
            order = calibration_rng(dataset, client_id, calibration_seed).permutation(expected)
        else:
            order = np.arange(expected, dtype=np.int64)

        cursor = 0
        counts = (
            (DataRole.REFERENCE, config.split.reference_benign),
            (DataRole.MISMATCH, config.split.mismatch_benign),
            (DataRole.CALIBRATION, config.split.calibration_benign),
            (DataRole.BENIGN_GUARD, config.split.benign_guard),
        )
        positions: dict[DataRole, tuple[int, ...]] = {}
        row_hashes: dict[DataRole, Sha256] = {}
        for role, count in counts:
            role_positions = tuple(int(value) for value in order[cursor : cursor + count])
            positions[role] = role_positions
            row_ids = reservoir.iloc[list(role_positions)]["row_id"].astype(str).tolist()
            row_hashes[role] = Sha256(hash_row_ids(row_ids))
            cursor += count
        if cursor != expected:
            raise RuntimeError("Calibration roles do not exactly cover the reservoir")
        flattened = [value for role in _CALIBRATION_ROLES for value in positions[role]]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(range(expected)):
            raise RuntimeError("Calibration assignment is not a partition of the reservoir")
        return CalibrationRoleAssignment(
            client_id=client_id,
            calibration_seed=calibration_seed,
            mode=mode,
            positions=positions,
            row_id_hashes=row_hashes,
        )

    @staticmethod
    def materialize(
        reservoir: pd.DataFrame,
        assignment: CalibrationRoleAssignment,
    ) -> dict[DataRole, pd.DataFrame]:
        return {
            role: reservoir.iloc[list(assignment.positions_for(role))].copy()
            for role in _CALIBRATION_ROLES
        }


class DataSplitter:
    """Build the seed-independent base partition and deterministic assignment views."""

    def __init__(self, assignment_builder: CalibrationAssignmentBuilder | None = None) -> None:
        self.assignments = assignment_builder or CalibrationAssignmentBuilder()

    def split_base(self, data: ClientData, config: DatasetConfig) -> ClientSplits:
        split = config.split
        benign = _ensure_row_ids(data.benign, data.dataset, data.client_id, "benign")
        attack = _ensure_row_ids(data.attack, data.dataset, data.client_id, "attack")
        if len(benign) < split.minimum_benign_rows:
            raise DataIntegrityError(
                f"Client {data.client_id} has {len(benign)} benign rows; "
                f"{split.minimum_benign_rows} required"
            )

        train_end = split.train_benign
        reservoir_end = train_end + split.reservoir_size
        roles: dict[DataRole, pd.DataFrame] = {
            DataRole.TRAIN: benign.iloc[:train_end].copy(),
            DataRole.RESERVOIR: benign.iloc[train_end:reservoir_end].copy(),
            DataRole.BENIGN_TEST: benign.iloc[reservoir_end:].copy(),
        }
        attack_dev, attack_test = self._attack_roles(attack, data, config)
        roles[DataRole.ATTACK_DEV] = attack_dev
        roles[DataRole.ATTACK_TEST] = attack_test
        for role, frame in roles.items():
            frame["role"] = role.value

        result = ClientSplits(data.client_id, roles)
        validate_split_disjointness(result)
        self._validate_base_counts(result, config)
        return result

    def calibration_assignment(
        self,
        base_splits: ClientSplits,
        dataset: DatasetId,
        config: DatasetConfig,
        calibration_seed: int,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> CalibrationRoleAssignment:
        return self.assignments.build(
            base_splits.get(DataRole.RESERVOIR),
            dataset,
            base_splits.client_id,
            config,
            calibration_seed,
            mode,
        )

    def materialize_calibration_roles(
        self,
        base_splits: ClientSplits,
        dataset: DatasetId,
        config: DatasetConfig,
        calibration_seed: int,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> dict[DataRole, pd.DataFrame]:
        assignment = self.calibration_assignment(
            base_splits, dataset, config, calibration_seed, mode
        )
        return self.assignments.materialize(
            base_splits.get(DataRole.RESERVOIR), assignment
        )

    @staticmethod
    def _attack_roles(
        attack: pd.DataFrame,
        data: ClientData,
        config: DatasetConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        split = config.split
        counts = attack_group_counts(attack)
        if data.dataset is DatasetId.NBAIOT:
            allocation = allocate_nbaiot_attack_development(
                counts,
                split.attack_dev,
                split.min_attack_test_per_group,
            )
        elif data.dataset is DatasetId.DIAD:
            if len(attack) < split.attack_dev + split.min_attack_test:
                raise DataIntegrityError(FailureCode.MALICIOUS_COUNT_LT_1000.value)
            allocation = allocate_diad_attack_development(
                counts,
                split.attack_dev,
                split.min_attack_test_per_group,
            )
        else:
            raise DataIntegrityError(f"No split contract for {data.dataset.value}")

        development, test = _sample_attack_development(
            attack,
            data.dataset,
            data.client_id,
            allocation,
            ATTACK_DEVELOPMENT_SEED,
        )
        if len(development) != split.attack_dev or len(test) < split.min_attack_test:
            code = (
                FailureCode.NBAIOT_ATTACK_BUDGET_FAIL
                if data.dataset is DatasetId.NBAIOT
                else FailureCode.ATTACK_DEV_CAPACITY_LT_500
            )
            raise DataIntegrityError(code.value)
        final_counts = attack_group_counts(test)
        for group, count in counts.items():
            reserve = min(split.min_attack_test_per_group, count)
            if final_counts.get(group, 0) < reserve:
                raise DataIntegrityError(
                    f"Final attack test lost required evidence for {group}"
                )
        return development, test

    @staticmethod
    def _validate_base_counts(splits: ClientSplits, config: DatasetConfig) -> None:
        expected = {
            DataRole.TRAIN: config.split.train_benign,
            DataRole.RESERVOIR: config.split.reservoir_size,
            DataRole.ATTACK_DEV: config.split.attack_dev,
        }
        for role, count in expected.items():
            if len(splits.get(role)) != count:
                raise DataIntegrityError(
                    f"Role {role.value} has {len(splits.get(role))} rows; expected {count}"
                )
        if len(splits.get(DataRole.BENIGN_TEST)) < config.split.min_benign_test:
            raise DataIntegrityError("Final benign test count is below the locked minimum")
        if len(splits.get(DataRole.ATTACK_TEST)) < config.split.min_attack_test:
            raise DataIntegrityError("Final attack test count is below the locked minimum")
