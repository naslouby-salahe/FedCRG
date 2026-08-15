from __future__ import annotations

import numpy as np
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
_CALIBRATION_ROLES = frozenset(
    {
        DataRole.REFERENCE,
        DataRole.MISMATCH,
        DataRole.CALIBRATION,
        DataRole.BENIGN_GUARD,
    }
)


class RoleFrame(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    role: DataRole
    frame: pd.DataFrame


class ClientSplits(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    client_id: ClientId
    roles: tuple[RoleFrame, ...]

    def get(self, role: DataRole) -> pd.DataFrame:
        frame = self.try_get(role)
        if frame is None:
            raise KeyError(role)
        return frame

    def try_get(self, role: DataRole) -> pd.DataFrame | None:
        return next((item.frame for item in self.roles if item.role is role), None)


class RolePositions(BaseModel):
    model_config = Frozen

    role: DataRole
    positions: tuple[Position, ...]
    row_id_hash: Sha256


class CalibrationRoleAssignment(BaseModel):
    model_config = Frozen

    client_id: ClientId
    calibration_seed: CalibrationSeed
    mode: CalibrationAssignmentMode
    roles: tuple[RolePositions, ...]

    def positions_for(self, role: DataRole) -> tuple[Position, ...]:
        if role not in _CALIBRATION_ROLES:
            raise ValueError(f"{role} is not a calibration-reservoir role")
        match = next((item.positions for item in self.roles if item.role is role), None)
        if match is None:
            raise KeyError(role)
        return match

    def row_id_hash_for(self, role: DataRole) -> Sha256:
        match = next((item.row_id_hash for item in self.roles if item.role is role), None)
        if match is None:
            raise KeyError(role)
        return match


def validate_split_disjointness(
    splits: ClientSplits,
    row_id_column: PreparedColumn = PreparedColumn.ROW_ID,
) -> None:
    """Every role's rows must be pairwise disjoint; a leaked row would invalidate downstream statistical guarantees."""
    seen: set[RowId] = set()
    for role in DataRole:
        frame = splits.try_get(role)
        if frame is None:
            continue
        if row_id_column not in frame.columns:
            raise DataIntegrityError(f"{role} is missing {row_id_column}")

        role_ids = set(frame[row_id_column].astype(str).unique())
        overlap = seen.intersection(role_ids)
        if overlap:
            raise DataIntegrityError(f"Split overlap in {role}: {sorted(overlap)[:5]}")
        seen.update(role_ids)


class CalibrationAssignmentBuilder:
    def build(
        self,
        frame: pd.DataFrame,
        dataset: DatasetId,
        client_id: ClientId,
        config: DatasetConfig,
        calibration_seed: CalibrationSeed,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> CalibrationRoleAssignment:
        """Splits the calibration reservoir into reference/mismatch/calibration/guard roles from a seeded permutation (or, in SOURCE_ORDER mode, the identity order) so each role sees only its own disjoint slice."""
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

        if mode is CalibrationAssignmentMode.SEEDED_PERMUTATION:
            rng = calibration_rng(dataset, client_id, calibration_seed)
            positions = tuple(int(index) for index in rng.permutation(reservoir_total))
        else:
            positions = tuple(range(reservoir_total))

        boundaries = (
            split.reference_benign,
            split.reference_benign + split.mismatch_benign,
            split.reference_benign + split.mismatch_benign + split.calibration_benign,
        )

        row_id_series = frame[PreparedColumn.ROW_ID].astype(str)
        roles: list[RolePositions] = []

        role_slices = (
            (DataRole.REFERENCE, 0, boundaries[0]),
            (DataRole.MISMATCH, boundaries[0], boundaries[1]),
            (DataRole.CALIBRATION, boundaries[1], boundaries[2]),
            (DataRole.BENIGN_GUARD, boundaries[2], reservoir_total),
        )

        for role, start, end in role_slices:
            role_positions = positions[start:end]
            row_ids = [str(row_id_series.iat[index]) for index in role_positions]
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
    model_config = Frozen

    group: AttackGroupId
    count: NonNegativeCount


class AttackGroupAllocation(BaseModel):
    model_config = Frozen

    groups: tuple[AttackGroupCount, ...]

    def for_group(self, group: AttackGroupId) -> NonNegativeCount:
        return next((item.count for item in self.groups if item.group == group), 0)


class BaseSplitBuilder:
    def build(
        self,
        data: ClientData,
        config: DatasetConfig,
        attack_split_seed: RngSeed,
    ) -> ClientSplits:
        """Consumes benign rows in source order — train, then the calibration reservoir, then benign test — so later roles never influence the fitted training boundary."""
        split = config.split
        benign = data.benign.reset_index(drop=True)

        req_benign = split.train_benign + split.reservoir_size + split.min_benign_test
        if len(benign) < req_benign:
            raise DataIntegrityError(f"Benign evidence is insufficient for {data.client_id}")

        train = benign.iloc[: split.train_benign].copy()
        reservoir = benign.iloc[
            split.train_benign : split.train_benign + split.reservoir_size
        ].copy()
        benign_test = benign.iloc[split.train_benign + split.reservoir_size : req_benign].copy()

        attack = data.attack.reset_index(drop=True)
        if PreparedColumn.ATTACK_GROUP not in attack.columns:
            raise DataIntegrityError(f"Attack frame lacks attack_group for {data.client_id}")

        group_values = attack[PreparedColumn.ATTACK_GROUP].astype(str)
        unique_groups = group_values.unique()
        groups = tuple(sorted(_ATTACK_GROUP_ADAPTER.validate_python(g) for g in unique_groups))

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
            members = np.flatnonzero(group_values == group)
            rng = attack_rng(data.dataset, data.client_id, group, attack_split_seed)
            chosen = rng.choice(len(members), size=development.for_group(group), replace=False)
            dev_rows.extend(int(members[index]) for index in chosen)

        # Everything not drawn into the fixed development budget is final-test-only attack data.
        dev_index = set(dev_rows)
        test_rows = [i for i in range(len(attack)) if i not in dev_index]

        attack_dev = attack.iloc[dev_rows].copy()
        attack_test = attack.iloc[test_rows].copy()

        role_frames = (
            (train, DataRole.TRAIN),
            (reservoir, DataRole.RESERVOIR),
            (benign_test, DataRole.BENIGN_TEST),
            (attack_dev, DataRole.ATTACK_DEV),
            (attack_test, DataRole.ATTACK_TEST),
        )

        processed_roles = []
        for frame, role in role_frames:
            frame[PreparedColumn.ROLE] = role
            frame[PreparedColumn.LABEL] = (
                0 if role in {DataRole.TRAIN, DataRole.RESERVOIR, DataRole.BENIGN_TEST} else 1
            )
            if PreparedColumn.ROW_ID not in frame.columns:
                frame[PreparedColumn.ROW_ID] = [
                    stable_row_id(data.dataset, data.client_id, role, int(i))
                    for i in range(len(frame))
                ]
            processed_roles.append(RoleFrame(role=role, frame=frame))

        splits = ClientSplits(client_id=data.client_id, roles=tuple(processed_roles))
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
        """DIAD's uneven attack-group sizes get a capacity-aware waterfill; N-BaIoT's near-even groups get an equal split with hard failure on shortfall."""
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
        if not groups:
            raise DataIntegrityError("Attack frame contains no attack groups")

        per_group, remainder = divmod(int(development_budget), len(groups))
        allocation = {group: int(per_group) for group in groups}

        for group in groups[:remainder]:
            allocation[group] += 1

        for group in groups:
            total = group_counts.for_group(group)
            capacity = total - min(reserve_per_group, total)
            if allocation[group] > capacity:
                raise DataIntegrityError(
                    f"{FailureCode.NBAIOT_ATTACK_BUDGET_FAIL}: attack group {group} "
                    f"cannot retain {reserve_per_group} final-test rows"
                )

        if sum(allocation.values()) != int(development_budget):
            raise RuntimeError("Attack development allocation is unbalanced")

        return AttackGroupAllocation(
            groups=tuple(AttackGroupCount(group=g, count=c) for g, c in allocation.items())
        )

    @staticmethod
    def waterfill_allocation(
        groups: tuple[AttackGroupId, ...],
        capacities: AttackGroupAllocation,
        development_budget: PositiveCount,
    ) -> AttackGroupAllocation:
        """Assigns the development budget one row at a time to whichever under-capacity group has the fewest rows so far, balancing across category/subtype without exceeding any group's capacity."""
        if not groups:
            raise DataIntegrityError("Attack development allocation requires groups")

        development = dict.fromkeys(groups, 0)
        budget = int(development_budget)

        for _ in range(budget):
            eligible = [g for g in groups if development[g] < capacities.for_group(g)]
            if not eligible:
                raise DataIntegrityError(
                    f"{FailureCode.ATTACK_DEV_CAPACITY_LT_500}: "
                    "attack development capacity is exhausted before the budget is met"
                )

            minimum = min(development[g] for g in eligible)
            chosen = next(g for g in eligible if development[g] == minimum)
            development[chosen] += 1

        for group in groups:
            if not 0 <= development[group] <= capacities.for_group(group):
                raise RuntimeError("Attack development allocation violates group capacity")

        return AttackGroupAllocation(
            groups=tuple(AttackGroupCount(group=g, count=c) for g, c in development.items())
        )
