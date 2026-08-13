"""Typed persistence for frozen preprocessing evidence."""

from __future__ import annotations

import json
from pathlib import Path

from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.core.enums import DatasetId
from fedcrg.core.ids import ClientId, Sha256
from fedcrg.data.preprocessing import (
    ClientPreprocessingParameters,
    PreprocessingModel,
)


class PreprocessingManifestStore:
    """Persist and validate the exact train-only preprocessing contract."""

    def save(self, path: Path, model: PreprocessingModel) -> None:
        atomic_write_json(path, model)

    def load(self, path: Path) -> PreprocessingModel:
        raw = json.loads(path.read_text(encoding="utf-8"))
        clients = tuple(
            ClientPreprocessingParameters(
                client_id=ClientId(str(item["client_id"])),
                training_row_sha256=Sha256(str(item["training_row_sha256"])),
                medians=(
                    None
                    if item["medians"] is None
                    else tuple(float(value) for value in item["medians"])
                ),
            )
            for item in raw["clients"]
        )
        model = PreprocessingModel(
            dataset=DatasetId(str(raw["dataset"])),
            feature_columns=tuple(str(value) for value in raw["feature_columns"]),
            clients=clients,
            global_minima=tuple(float(value) for value in raw["global_minima"]),
            global_maxima=tuple(float(value) for value in raw["global_maxima"]),
        )
        expected_constant = tuple(bool(value) for value in raw["constant_features"])
        if model.constant_features != expected_constant:
            raise ValueError("Preprocessing constant-feature ledger mismatch")
        if int(raw["extrema_upload_bytes_per_client"]) != model.extrema_upload_bytes_per_client:
            raise ValueError("Preprocessing per-client communication ledger mismatch")
        if int(raw["extrema_upload_bytes_total"]) != model.extrema_upload_bytes_total:
            raise ValueError("Preprocessing total communication ledger mismatch")
        return model
