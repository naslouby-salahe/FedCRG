"""DIAD R14 training-schema-only numeric-safe feature derivation."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np
import pandas as pd

from fedcrg.data.prepare import hash_row_ids
from fedcrg.domain.identifiers import ClientId, Sha256

# Exclude direct identity/label/port fields, not behavioral statistics whose names
# happen to contain terms such as "stream" or "mac".
_EXCLUDED_EXACT = {
    "stream",
    "device_mac",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "port_class_dst",
    "label",
    "Label",
    "anomaly",
    "is_anomaly",
    "attack",
    "row_id",
    "source_file",
    "source_row_index",
    "capture_time",
    "verified_chronology",
    "attack_group",
    "role",
}
_EXCLUDED_NAME_MARKERS = (
    "user_agent",
    "hostname",
    "domain_name",
    "uri",
    "mac_address",
    "ip_address",
)


@dataclass(frozen=True, slots=True)
class ClientTrainingRowHash:
    client_id: ClientId
    sha256: Sha256


@dataclass(frozen=True, slots=True)
class NumericSafeFeatureContract:
    features: tuple[str, ...]
    dimension: int
    architecture: tuple[int, ...]
    training_row_hashes: tuple[ClientTrainingRowHash, ...]

    @property
    def encoder_hidden_dims(self) -> tuple[int, ...]:
        return self.architecture[1:5]


def _is_forbidden_name(column: str) -> bool:
    lowered = column.lower()
    if column in _EXCLUDED_EXACT or lowered in {
        value.lower() for value in _EXCLUDED_EXACT
    }:
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
        if column.startswith("_") or _is_forbidden_name(column):
            continue
        if not all(
            pd.api.types.is_numeric_dtype(frame[column])
            for frame in training_frames.values()
        ):
            continue
        finite_for_all = True
        for frame in training_frames.values():
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            if float(np.isfinite(values).mean()) < 0.99:
                finite_for_all = False
                break
        if finite_for_all:
            selected.append(column)

    dimension = len(selected)
    if dimension == 0:
        raise ValueError("R14 numeric-safe feature derivation produced no features")
    architecture = (
        dimension,
        max(1, floor(0.75 * dimension)),
        max(1, floor(0.50 * dimension)),
        max(1, floor(dimension / 3)),
        max(1, floor(0.25 * dimension)),
        max(1, floor(dimension / 3)),
        max(1, floor(0.50 * dimension)),
        max(1, floor(0.75 * dimension)),
        dimension,
    )
    training_hashes: list[ClientTrainingRowHash] = []
    for client, frame in sorted(training_frames.items()):
        if "row_id" not in frame.columns:
            raise ValueError(f"R14 training frame is missing row_id for {client.value}")
        training_hashes.append(
            ClientTrainingRowHash(
                client,
                hash_row_ids(frame["row_id"].astype(str).tolist()),
            )
        )
    return NumericSafeFeatureContract(
        features=tuple(selected),
        dimension=dimension,
        architecture=architecture,
        training_row_hashes=tuple(training_hashes),
    )
