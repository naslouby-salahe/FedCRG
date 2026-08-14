"""Deterministic split construction: base role cutting, attack-development
allocation, calibration-role assignment, and split validation.

Base roles (train/reservoir/benign_test/attack_dev/attack_test) are cut from
one client's benign and attack frames by fixed positional slices plus a seeded
attack-development draw. Calibration roles are a seeded permutation of the
reservoir. Every stage validates row-id disjointness before returning.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, TypeAdapter

from fedcrg.config import DatasetConfig
from fedcrg.data.datasets import (
    ClientData,
    attack_rng,
    calibration_rng,
    hash_row_ids,
    stable_row_id,
)
from fedcrg.types import (
    AttackGroupId,
    CalibrationAssignmentMode,
    CalibrationSeed,
    ClientId,
    DataIntegrityError,
    DataRole,
    DatasetId,
    FailureCode,
    NonNegativeCount,
    PositiveCount,
    Position,
    PreparedColumn,
    RngSeed,
    RowId,
    Sha256,
)

Frozen = ConfigDict(frozen=True)
_ATTACK_GROUP_ADAPTER = TypeAdapter(AttackGroupId)


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
    seen: set[RowId] = set()
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


class AttackGroupCount(BaseModel):
    """One attack group's row count."""

    model_config = Frozen

    group: AttackGroupId
    count: NonNegativeCount


class AttackGroupAllocation(BaseModel):
    """Per-attack-group row allocation, keyed by attack group identity."""

    model_config = Frozen

    groups: tuple[AttackGroupCount, ...]

    def for_group(self, group: AttackGroupId) -> NonNegativeCount:
        for item in self.groups:
            if item.group == group:
                return item.count
        return 0


class BaseSplitBuilder:
    """Cut deterministic base roles from one client's benign/attack frames."""

    def build(
        self,
        data: ClientData,
        config: DatasetConfig,
        attack_split_seed: RngSeed,
    ) -> ClientSplits:
        split = config.split
        benign = data.benign.reset_index(drop=True)
        if len(benign) < split.train_benign + split.reservoir_size + split.min_benign_test:
            raise DataIntegrityError(f"Benign evidence is insufficient for {data.client_id}")
        train = benign.iloc[: split.train_benign].copy()
        reservoir = benign.iloc[
            split.train_benign : split.train_benign + split.reservoir_size
        ].copy()
        benign_test = benign.iloc[
            split.train_benign + split.reservoir_size : split.train_benign
            + split.reservoir_size
            + split.min_benign_test
        ].copy()
        attack = data.attack.reset_index(drop=True)
        if PreparedColumn.ATTACK_GROUP.value not in attack.columns:
            raise DataIntegrityError(f"Attack frame lacks attack_group for {data.client_id}")
        group_values = attack[PreparedColumn.ATTACK_GROUP.value].astype(str)
        groups = tuple(
            sorted(_ATTACK_GROUP_ADAPTER.validate_python(value) for value in set(group_values))
        )
        group_counts = AttackGroupAllocation(
            groups=tuple(
                AttackGroupCount(
                    group=_ATTACK_GROUP_ADAPTER.validate_python(str(group)), count=int(count)
                )
                for group, count in group_values.value_counts().items()
            )
        )
        development = self._development_allocation(
            data.dataset,
            groups,
            group_counts,
            split.attack_dev,
            split.min_attack_test_per_group,
        )
        dev_rows: list[Position] = []
        for group in groups:
            members = sorted(
                index for index in range(len(attack)) if group_values.iloc[index] == group
            )
            rng = attack_rng(data.dataset, data.client_id, group, attack_split_seed)
            chosen = tuple(
                int(index)
                for index in rng.choice(
                    len(members), size=development.for_group(group), replace=False
                )
            )
            dev_rows.extend(members[index] for index in chosen)
        dev_index = set(dev_rows)
        test_rows = [index for index in range(len(attack)) if index not in dev_index]
        attack_dev = attack.iloc[dev_rows].copy()
        attack_test = attack.iloc[test_rows].copy()

        for frame, role in (
            (train, DataRole.TRAIN),
            (reservoir, DataRole.RESERVOIR),
            (benign_test, DataRole.BENIGN_TEST),
            (attack_dev, DataRole.ATTACK_DEV),
            (attack_test, DataRole.ATTACK_TEST),
        ):
            frame[PreparedColumn.ROLE.value] = role.value
            frame[PreparedColumn.LABEL.value] = (
                0 if role in {DataRole.TRAIN, DataRole.RESERVOIR, DataRole.BENIGN_TEST} else 1
            )
            if PreparedColumn.ROW_ID.value not in frame.columns:
                frame[PreparedColumn.ROW_ID.value] = [
                    stable_row_id(data.dataset, data.client_id, role.value, int(index))
                    for index in range(len(frame))
                ]
        splits = ClientSplits(
            client_id=data.client_id,
            roles=tuple(
                RoleFrame(role=role, frame=frame)
                for role, frame in (
                    (DataRole.TRAIN, train),
                    (DataRole.RESERVOIR, reservoir),
                    (DataRole.BENIGN_TEST, benign_test),
                    (DataRole.ATTACK_DEV, attack_dev),
                    (DataRole.ATTACK_TEST, attack_test),
                )
            ),
        )
        validate_split_disjointness(splits)
        return splits

    @staticmethod
    def _development_allocation(
        dataset: DatasetId,
        groups: tuple[AttackGroupId, ...],
        group_counts: AttackGroupAllocation,
        development_budget: PositiveCount,
        reserve_per_group: PositiveCount,
    ) -> AttackGroupAllocation:
        """Per-group development counts under the dataset's locked allocation rule."""
        if dataset is DatasetId.DIAD:
            capacities = AttackGroupAllocation(
                groups=tuple(
                    AttackGroupCount(
                        group=item.group,
                        count=max(0, item.count - min(reserve_per_group, item.count)),
                    )
                    for item in group_counts.groups
                )
            )
            return BaseSplitBuilder.waterfill_allocation(groups, capacities, development_budget)
        return BaseSplitBuilder._nbaiot_balanced_allocation(
            groups, group_counts, development_budget, reserve_per_group
        )

    @staticmethod
    def _nbaiot_balanced_allocation(
        groups: tuple[AttackGroupId, ...],
        group_counts: AttackGroupAllocation,
        development_budget: PositiveCount,
        reserve_per_group: PositiveCount,
    ) -> AttackGroupAllocation:
        """Equal floor allocation with the remainder distributed lexicographically.

        Every present attack group receives ``floor(budget / m)`` development
        records, then the remainder is distributed one record at a time in
        lexicographic group order. A group that cannot retain the minimum
        final-test evidence after that allocation blocks the run.
        """
        if not groups:
            raise DataIntegrityError("Attack frame contains no attack groups")
        per_group, remainder = divmod(int(development_budget), len(groups))
        allocation: dict[AttackGroupId, NonNegativeCount] = {
            group: int(per_group) for group in groups
        }
        for group in groups[:remainder]:
            allocation[group] = int(allocation[group]) + 1
        for group in groups:
            total = group_counts.for_group(group)
            capacity = total - min(reserve_per_group, total)
            if int(allocation[group]) > capacity:
                raise DataIntegrityError(
                    f"{FailureCode.NBAIOT_ATTACK_BUDGET_FAIL.value}: attack group {group} "
                    f"cannot retain {reserve_per_group} final-test rows"
                )
        if sum(int(value) for value in allocation.values()) != int(development_budget):
            raise RuntimeError("Attack development allocation is unbalanced")
        return AttackGroupAllocation(
            groups=tuple(AttackGroupCount(group=group, count=allocation[group]) for group in groups)
        )

    @staticmethod
    def waterfill_allocation(
        groups: tuple[AttackGroupId, ...],
        capacities: AttackGroupAllocation,
        development_budget: PositiveCount,
    ) -> AttackGroupAllocation:
        """Most-even per-group allocation bounded by each group's capacity.

        At every step the lexicographically first group with the current
        minimum development count below its capacity receives one record. The
        result is the most even deterministic allocation possible subject to
        the per-group capacity constraints.
        """
        if not groups:
            raise DataIntegrityError("Attack development allocation requires groups")
        development: dict[AttackGroupId, NonNegativeCount] = {group: 0 for group in groups}
        for _ in range(int(development_budget)):
            eligible = [
                group
                for group in groups
                if int(development[group]) < int(capacities.for_group(group))
            ]
            if not eligible:
                raise DataIntegrityError(
                    f"{FailureCode.ATTACK_DEV_CAPACITY_LT_500.value}: "
                    "attack development capacity is exhausted before the budget is met"
                )
            minimum = min(int(development[group]) for group in eligible)
            chosen = next(group for group in eligible if int(development[group]) == minimum)
            development[chosen] = int(development[chosen]) + 1
        for group in groups:
            if not 0 <= int(development[group]) <= int(capacities.for_group(group)):
                raise RuntimeError("Attack development allocation violates group capacity")
        return AttackGroupAllocation(
            groups=tuple(
                AttackGroupCount(group=group, count=development[group]) for group in groups
            )
        )
