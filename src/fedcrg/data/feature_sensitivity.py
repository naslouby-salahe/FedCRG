"""DIAD R14 training-schema-only numeric-safe feature derivation."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np
import pandas as pd

from fedcrg.core.ids import ClientId, Sha256
from fedcrg.data.manifests import hash_row_ids

_EXCLUDED_TOKENS = (
    "label",
    "device_mac",
    "mac",
    "src_ip",
    "dst_ip",
    "ip_address",
    "port",
    "uri",
    "user_agent",
    "hostname",
    "domain",
    "oui",
    "tls",
    "http",
    "dns",
    "stream",
)
_METADATA_COLUMNS = {
    "_source_file",
    "_source_row_index",
    "_row_id",
    "_capture_time",
}


@dataclass(frozen=True, slots=True)
class NumericSafeFeatureContract:
    features: tuple[str, ...]
    dimension: int
    architecture: tuple[int, ...]
    training_row_hashes: dict[ClientId, Sha256]

    @property
    def encoder_hidden_dims(self) -> tuple[int, ...]:
        return self.architecture[1:5]

    def to_dict(self) -> dict[str, object]:
        return {
            "features": list(self.features),
            "dimension": self.dimension,
            "architecture": list(self.architecture),
            "training_row_hashes": {
                client.value: digest.value
                for client, digest in sorted(self.training_row_hashes.items())
            },
        }


def derive_numeric_safe_features(
    training_frames: dict[ClientId, pd.DataFrame],
) -> NumericSafeFeatureContract:
    """Derive R14 features using training rows only and no outcome association."""
    if not training_frames:
        raise ValueError("R14 requires training frames")
    common = set.intersection(*(set(frame.columns) for frame in training_frames.values()))
    selected: list[str] = []
    for column in sorted(common):
        if column in _METADATA_COLUMNS or column.startswith("_"):
            continue
        lowered = column.lower()
        if any(token in lowered for token in _EXCLUDED_TOKENS):
            continue
        if not all(
            pd.api.types.is_numeric_dtype(frame[column])
            for frame in training_frames.values()
        ):
            continue
        finite_for_all = True
        for frame in training_frames.values():
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            values[~np.isfinite(values)] = np.nan
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
    training_hashes = {
        client: Sha256(hash_row_ids(frame["_row_id"].astype(str).tolist()))
        for client, frame in training_frames.items()
    }
    return NumericSafeFeatureContract(
        features=tuple(selected),
        dimension=dimension,
        architecture=architecture,
        training_row_hashes=training_hashes,
    )
