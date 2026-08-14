"""Frozen train-only preprocessing: per-client statistics, aggregation, and
the fitted transform applied at materialization time.

Statistics are computed locally per client on the train role only, then only
the train-set extrema are aggregated globally, so no client's raw feature
values leave its own computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from fedcrg.data.datasets import hash_row_ids
from fedcrg.data.splits import ClientSplits
from fedcrg.types import (
    ClientId,
    DataIntegrityError,
    DataRole,
    DatasetId,
    FeatureName,
    PositiveCount,
    PreparedColumn,
    Score,
    Sha256,
)

Frozen = ConfigDict(frozen=True)

_METADATA = {column.value for column in PreparedColumn}
_DIAD_FINITE_RATE_MINIMUM = 0.99


def model_feature_columns(frame: pd.DataFrame, expected: PositiveCount) -> tuple[FeatureName, ...]:
    """Resolve the model-feature column tuple for one dataset."""
    columns = tuple(column for column in frame.columns if column not in _METADATA)
    if len(columns) != expected:
        raise DataIntegrityError(f"Expected {expected} model features, found {len(columns)}")
    return columns


class ClientPreprocessingParameters(BaseModel):
    """Frozen per-client preprocessing parameterization."""

    model_config = Frozen

    client_id: ClientId
    training_row_sha256: Sha256
    medians: tuple[Score, ...] | None = None


class PreprocessingModel(BaseModel):
    """Frozen train-only preprocessing model."""

    model_config = Frozen

    dataset: DatasetId
    feature_columns: tuple[FeatureName, ...]
    clients: tuple[ClientPreprocessingParameters, ...]
    global_minima: tuple[Score, ...]
    global_maxima: tuple[Score, ...]

    @property
    def constant_features(self) -> tuple[bool, ...]:
        """Per-feature flag of zero global span (M == m), recorded for the audit."""
        return tuple(
            minimum == maximum
            for minimum, maximum in zip(self.global_minima, self.global_maxima, strict=True)
        )

    def parameters_for(self, client_id: ClientId) -> ClientPreprocessingParameters:
        for item in self.clients:
            if item.client_id == client_id:
                return item
        raise KeyError(client_id)

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
            raise DataIntegrityError(
                f"Non-finite values remain after preprocessing for {client_id}"
            )
        minima = np.asarray(self.global_minima, dtype=np.float64)
        maxima = np.asarray(self.global_maxima, dtype=np.float64)
        span = maxima - minima
        scaled = np.zeros_like(values, dtype=np.float64)
        varying = span != 0.0
        scaled[:, varying] = (values[:, varying] - minima[varying]) / span[varying]
        result = frame.copy()
        result.loc[:, list(self.feature_columns)] = scaled
        return result


class ClientPreprocessingStatistics(BaseModel):
    """Frozen per-client preprocessing statistics."""

    model_config = Frozen

    client_id: ClientId
    feature_columns: tuple[FeatureName, ...]
    training_row_hash: Sha256
    local_minima: tuple[Score, ...]
    local_maxima: tuple[Score, ...]
    medians: tuple[Score, ...] | None = None


class TrainOnlyPreprocessing:
    """Compute per-client statistics locally, then aggregate only train-set extrema."""

    def validate_training_rows(
        self,
        splits: ClientSplits,
        dataset: DatasetId,
        expected_features: PositiveCount,
    ) -> tuple[FeatureName, ...]:
        train = splits.get(DataRole.TRAIN)
        columns = model_feature_columns(train, expected_features)
        values = train.loc[:, list(columns)].to_numpy(dtype=np.float64)
        if dataset is DatasetId.NBAIOT:
            if not np.isfinite(values).all():
                raise DataIntegrityError("N-BaIoT training features must all be finite")
        elif dataset is DatasetId.DIAD:
            finite_rate = np.isfinite(values).mean(axis=0)
            failing = [
                columns[index]
                for index, rate in enumerate(finite_rate)
                if rate < _DIAD_FINITE_RATE_MINIMUM
            ]
            if failing:
                raise DataIntegrityError("DIAD_FEATURE_FINITE_RATE_FAIL: " + ", ".join(failing[:5]))
        return columns

    def client_statistics(
        self,
        splits: ClientSplits,
        dataset: DatasetId,
        expected_features: PositiveCount,
    ) -> ClientPreprocessingStatistics:
        columns = self.validate_training_rows(splits, dataset, expected_features)
        train_frame = splits.get(DataRole.TRAIN)
        training_row_hash = hash_row_ids(
            train_frame[PreparedColumn.ROW_ID.value].astype(str).tolist()
        )
        train_values = train_frame.loc[:, list(columns)].to_numpy(dtype=np.float64, copy=True)
        if dataset is DatasetId.DIAD:
            medians = np.nanmedian(
                np.where(np.isfinite(train_values), train_values, np.nan), axis=0
            )
            if not np.isfinite(medians).all():
                raise DataIntegrityError(
                    f"DIAD imputation median is undefined for {splits.client_id}"
                )
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
