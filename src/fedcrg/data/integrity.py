"""Dataset and split integrity validation."""

from __future__ import annotations

import numpy as np

from fedcrg.core.enums import DataRole
from fedcrg.core.exceptions import DataIntegrityError
from fedcrg.data.models import ClientSplits


def validate_numeric_frame(frame, feature_columns: list[str]) -> None:
    values = frame[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise DataIntegrityError("Feature matrix contains NaN or infinite values")


def validate_split_disjointness(splits: ClientSplits, row_id_column: str = "row_id") -> None:
    seen: set[str] = set()
    for role in DataRole:
        frame = splits.roles.get(role)
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
