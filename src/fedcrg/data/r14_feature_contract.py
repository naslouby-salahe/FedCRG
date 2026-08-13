"""R14 training-schema-only numeric-safe feature contract.

This implementation deliberately excludes only direct identity/label/port/application
fields. It does not reject behavioral statistics merely because their names contain
words such as ``stream`` or ``src_ip``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np
import pandas as pd

from fedcrg.core.ids import ClientId, Sha256
from fedcrg.data.manifests import hash_row_ids

_EXACT_EXCLUDED = {
    "stream",
    "device_mac",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "port_class_dst",
    "most_freq_spot",
    "label",
    "Label",
    "anomaly",
    "is_anomaly",
    "attack",
}
_IDENTITY_PREFIXES = (
    "tls_",
    "http_",
    "dns_",
    "oui_",
    "user_agent",
    "uri_",
)
_METADATA = {
    "row_id",
    "_row_id",
    "_source_file",
    "_source_row_index",
    "_capture_time",
    "_verified_chronology",
}


@dataclass(frozen=True, slots=True)
class R14FeatureContract:
    features: tuple[str, ...]
    dimension: int
    architecture: tuple[int, ...]
    training_row_hashes: dict[ClientId, Sha256]

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


def derive_r14_feature_contract(
    training_frames: dict[ClientId, pd.DataFrame],
) -> R14FeatureContract:
    """Freeze numeric-safe columns using eligible-client training rows only."""

    if not training_frames:
        raise ValueError("R14 requires training frames from eligible DIAD clients")
    common = set.intersection(*(set(frame.columns) for frame in training_frames.values()))
    selected: list[str] = []
    for column in sorted(common):
        lowered = column.lower()
        if column in _METADATA or column.startswith("_"):
            continue
        if column in _EXACT_EXCLUDED or lowered in {value.lower() for value in _EXACT_EXCLUDED}:
            continue
        if any(lowered.startswith(prefix) for prefix in _IDENTITY_PREFIXES):
            continue
        if not all(pd.api.types.is_numeric_dtype(frame[column]) for frame in training_frames.values()):
            continue
        finite_for_all = True
        for frame in training_frames.values():
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            finite_rate = float(np.isfinite(values).mean())
            if finite_rate < 0.99:
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
        client: Sha256(hash_row_ids(frame["row_id"].astype(str).tolist()))
        for client, frame in training_frames.items()
    }
    return R14FeatureContract(
        features=tuple(selected),
        dimension=dimension,
        architecture=architecture,
        training_row_hashes=training_hashes,
    )
