"""Frozen train-only preprocessing for federated detector inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fedcrg.data.prepare import hash_row_ids
from fedcrg.data.splits import ClientSplits
from fedcrg.domain.enums import DataRole, DatasetId
from fedcrg.domain.errors import DataIntegrityError
from fedcrg.domain.identifiers import ClientId, Sha256

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

# Matches the float64 serialization convention in fedcrg.analysis.communication.
_FLOAT64_BYTES = 8


def model_feature_columns(frame: pd.DataFrame, expected: int) -> tuple[str, ...]:
    columns = tuple(column for column in frame.columns if column not in _METADATA)
    if len(columns) != expected:
        raise DataIntegrityError(f"Expected {expected} model features, found {len(columns)}")
    return columns


@dataclass(frozen=True, slots=True)
class ClientPreprocessingParameters:
    """One client's frozen local imputation parameters and training-row identity."""

    client_id: ClientId
    training_row_sha256: Sha256
    medians: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class PreprocessingModel:
    dataset: DatasetId
    feature_columns: tuple[str, ...]
    clients: tuple[ClientPreprocessingParameters, ...]
    global_minima: tuple[float, ...]
    global_maxima: tuple[float, ...]

    @property
    def constant_features(self) -> tuple[bool, ...]:
        return tuple(low == high for low, high in zip(self.global_minima, self.global_maxima, strict=True))

    @property
    def extrema_upload_bytes_per_client(self) -> int:
        return 2 * len(self.feature_columns) * _FLOAT64_BYTES

    @property
    def extrema_upload_bytes_total(self) -> int:
        return self.extrema_upload_bytes_per_client * len(self.clients)

    def parameters_for(self, client_id: ClientId) -> ClientPreprocessingParameters:
        for item in self.clients:
            if item.client_id == client_id:
                return item
        raise KeyError(client_id.value)

    def transform(self, frame: pd.DataFrame, client_id: ClientId) -> pd.DataFrame:
        values = frame.loc[:, list(self.feature_columns)].to_numpy(dtype=np.float64, copy=True)
        parameters = self.parameters_for(client_id)
        if parameters.medians is not None:
            medians = np.asarray(parameters.medians, dtype=np.float64)
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
            "clients": [
                {
                    "client_id": item.client_id.value,
                    "training_row_sha256": item.training_row_sha256.value,
                    "medians": None if item.medians is None else list(item.medians),
                }
                for item in sorted(self.clients, key=lambda item: item.client_id)
            ],
            "global_minima": list(self.global_minima),
            "global_maxima": list(self.global_maxima),
            "constant_features": list(self.constant_features),
            "extrema_upload_bytes_per_client": self.extrema_upload_bytes_per_client,
            "extrema_upload_bytes_total": self.extrema_upload_bytes_total,
        }


@dataclass(frozen=True, slots=True)
class ClientPreprocessingStatistics:
    """One client's local training-row statistics, safe to retain after the raw frame is discarded."""

    client_id: ClientId
    feature_columns: tuple[str, ...]
    training_row_hash: Sha256
    local_minima: tuple[float, ...]
    local_maxima: tuple[float, ...]
    medians: tuple[float, ...] | None


class FederatedPreprocessor:
    """Compute per-client statistics locally, then aggregate only train-set feature extrema."""

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

    def client_statistics(
        self,
        splits: ClientSplits,
        dataset: DatasetId,
        expected_features: int,
    ) -> ClientPreprocessingStatistics:
        """Fit one client's local imputer and training-row extrema without a global pass."""

        columns = self.validate_training_rows(splits, dataset, expected_features)
        train_frame = splits.get(DataRole.TRAIN)
        training_row_hash = hash_row_ids(train_frame["row_id"].astype(str).tolist())
        train_values = train_frame.loc[:, list(columns)].to_numpy(dtype=np.float64, copy=True)
        if dataset is DatasetId.DIAD:
            medians = np.nanmedian(np.where(np.isfinite(train_values), train_values, np.nan), axis=0)
            if not np.isfinite(medians).all():
                raise DataIntegrityError(f"DIAD imputation median is undefined for {splits.client_id}")
            missing = ~np.isfinite(train_values)
            if missing.any():
                rows, indices = np.nonzero(missing)
                train_values[rows, indices] = medians[indices]
            median_tuple: tuple[float, ...] | None = tuple(float(value) for value in medians)
        else:
            median_tuple = None
        return ClientPreprocessingStatistics(
            client_id=splits.client_id,
            feature_columns=columns,
            training_row_hash=training_row_hash,
            local_minima=tuple(float(value) for value in np.min(train_values, axis=0)),
            local_maxima=tuple(float(value) for value in np.max(train_values, axis=0)),
            medians=median_tuple,
        )

    def aggregate(
        self,
        statistics: tuple[ClientPreprocessingStatistics, ...],
        dataset: DatasetId,
    ) -> PreprocessingModel:
        """Combine local client statistics into the federation-wide scaling contract."""

        if not statistics:
            raise DataIntegrityError("Cannot fit preprocessing without clients")
        ordered = tuple(sorted(statistics, key=lambda item: item.client_id))
        feature_columns = ordered[0].feature_columns
        for item in ordered[1:]:
            if item.feature_columns != feature_columns:
                raise DataIntegrityError("Model feature order differs across clients")
        global_minima = np.min(np.stack([item.local_minima for item in ordered]), axis=0)
        global_maxima = np.max(np.stack([item.local_maxima for item in ordered]), axis=0)
        return PreprocessingModel(
            dataset=dataset,
            feature_columns=feature_columns,
            clients=tuple(
                ClientPreprocessingParameters(
                    client_id=item.client_id,
                    training_row_sha256=item.training_row_hash,
                    medians=item.medians,
                )
                for item in ordered
            ),
            global_minima=tuple(float(value) for value in global_minima),
            global_maxima=tuple(float(value) for value in global_maxima),
        )
