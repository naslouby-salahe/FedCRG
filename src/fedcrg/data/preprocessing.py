"""Frozen train-only preprocessing for federated detector inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fedcrg.core.enums import DataRole, DatasetId
from fedcrg.core.exceptions import DataIntegrityError
from fedcrg.core.ids import ClientId, Sha256
from fedcrg.data.models import ClientSplits
from fedcrg.data.manifests import hash_row_ids

_METADATA = {
    "row_id",
    "role",
    "label",
    "attack_group",
    "_source_file",
    "_source_row_index",
    "_capture_time",
    "_verified_chronology",
    "source_file",
    "source_row_index",
    "capture_time",
}


def model_feature_columns(frame: pd.DataFrame, expected: int) -> tuple[str, ...]:
    columns = tuple(column for column in frame.columns if column not in _METADATA)
    if len(columns) != expected:
        raise DataIntegrityError(f"Expected {expected} model features, found {len(columns)}")
    return columns


@dataclass(frozen=True, slots=True)
class ClientImputer:
    medians: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class PreprocessingModel:
    dataset: DatasetId
    feature_columns: tuple[str, ...]
    client_imputers: dict[ClientId, ClientImputer]
    training_row_hashes: dict[ClientId, Sha256]
    global_minima: tuple[float, ...]
    global_maxima: tuple[float, ...]

    @property
    def constant_features(self) -> tuple[bool, ...]:
        return tuple(low == high for low, high in zip(self.global_minima, self.global_maxima, strict=True))

    def transform(self, frame: pd.DataFrame, client_id: ClientId) -> pd.DataFrame:
        values = frame.loc[:, list(self.feature_columns)].to_numpy(dtype=np.float64, copy=True)
        imputer = self.client_imputers[client_id]
        if imputer.medians is not None:
            medians = np.asarray(imputer.medians, dtype=np.float64)
            missing = ~np.isfinite(values)
            if missing.any():
                rows, columns = np.nonzero(missing)
                values[rows, columns] = medians[columns]
        if not np.isfinite(values).all():
            raise DataIntegrityError(f"Non-finite values remain after preprocessing for {client_id}")
        minima = np.asarray(self.global_minima, dtype=np.float64)
        maxima = np.asarray(self.global_maxima, dtype=np.float64)
        span = maxima - minima
        scaled = np.zeros_like(values, dtype=np.float64)
        varying = span != 0.0
        scaled[:, varying] = (values[:, varying] - minima[varying]) / span[varying]
        result = frame.copy()
        result.loc[:, list(self.feature_columns)] = scaled
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset.value,
            "feature_columns": list(self.feature_columns),
            "training_row_hashes": {client.value: hash_value.value for client, hash_value in sorted(self.training_row_hashes.items())},
            "client_medians": {
                client_id.value: None if item.medians is None else list(item.medians)
                for client_id, item in self.client_imputers.items()
            },
            "global_minima": list(self.global_minima),
            "global_maxima": list(self.global_maxima),
            "constant_features": list(self.constant_features),
        }


class FederatedPreprocessor:
    """Fit imputers locally and aggregate only train-set feature extrema."""

    def validate_training_rows(
        self,
        splits: ClientSplits,
        dataset: DatasetId,
        expected_features: int,
    ) -> tuple[str, ...]:
        train = splits.get(DataRole.TRAIN)
        columns = model_feature_columns(train, expected_features)
        values = train.loc[:, list(columns)].to_numpy(dtype=np.float64)
        if dataset is DatasetId.NBAIOT:
            if not np.isfinite(values).all():
                raise DataIntegrityError("N-BaIoT training features must all be finite")
        elif dataset is DatasetId.DIAD:
            finite_rate = np.isfinite(values).mean(axis=0)
            failing = [columns[index] for index, rate in enumerate(finite_rate) if rate < 0.99]
            if failing:
                raise DataIntegrityError(
                    "DIAD_FEATURE_FINITE_RATE_FAIL: " + ", ".join(failing[:5])
                )
        return columns

    def fit(
        self,
        splits_by_client: dict[ClientId, ClientSplits],
        dataset: DatasetId,
        expected_features: int,
    ) -> PreprocessingModel:
        if not splits_by_client:
            raise DataIntegrityError("Cannot fit preprocessing without clients")
        feature_columns: tuple[str, ...] | None = None
        imputers: dict[ClientId, ClientImputer] = {}
        training_row_hashes: dict[ClientId, Sha256] = {}
        local_minima: list[np.ndarray] = []
        local_maxima: list[np.ndarray] = []
        for client_id in sorted(splits_by_client):
            splits = splits_by_client[client_id]
            columns = self.validate_training_rows(splits, dataset, expected_features)
            if feature_columns is None:
                feature_columns = columns
            elif columns != feature_columns:
                raise DataIntegrityError("Model feature order differs across clients")
            train_frame = splits.get(DataRole.TRAIN)
            training_row_hashes[client_id] = Sha256(hash_row_ids(
                train_frame["row_id"].astype(str).tolist()
            ))
            train_values = train_frame.loc[:, list(columns)].to_numpy(
                dtype=np.float64, copy=True
            )
            if dataset is DatasetId.DIAD:
                medians = np.nanmedian(np.where(np.isfinite(train_values), train_values, np.nan), axis=0)
                if not np.isfinite(medians).all():
                    raise DataIntegrityError(f"DIAD imputation median is undefined for {client_id}")
                missing = ~np.isfinite(train_values)
                if missing.any():
                    rows, indices = np.nonzero(missing)
                    train_values[rows, indices] = medians[indices]
                imputers[client_id] = ClientImputer(tuple(float(value) for value in medians))
            else:
                imputers[client_id] = ClientImputer(None)
            local_minima.append(np.min(train_values, axis=0))
            local_maxima.append(np.max(train_values, axis=0))
        assert feature_columns is not None
        global_minima = np.min(np.stack(local_minima), axis=0)
        global_maxima = np.max(np.stack(local_maxima), axis=0)
        return PreprocessingModel(
            dataset=dataset,
            feature_columns=feature_columns,
            client_imputers=imputers,
            training_row_hashes=training_row_hashes,
            global_minima=tuple(float(value) for value in global_minima),
            global_maxima=tuple(float(value) for value in global_maxima),
        )
