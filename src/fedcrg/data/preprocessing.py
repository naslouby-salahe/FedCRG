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
    Probability,
    Score,
    Sha256,
)

Frozen = ConfigDict(frozen=True)

_METADATA = frozenset(PreparedColumn)


def model_feature_columns(frame: pd.DataFrame, expected: PositiveCount) -> tuple[FeatureName, ...]:
    columns = tuple(col for col in frame.columns if col not in _METADATA)
    if len(columns) != expected:
        raise DataIntegrityError(f"Expected {expected} model features, found {len(columns)}")
    return columns


class ClientPreprocessingParameters(BaseModel):
    model_config = Frozen

    client_id: ClientId
    training_row_sha256: Sha256
    medians: tuple[Score, ...] | None = None


class PreprocessingModel(BaseModel):
    model_config = Frozen

    dataset: DatasetId
    feature_columns: tuple[FeatureName, ...]
    clients: tuple[ClientPreprocessingParameters, ...]
    global_minima: tuple[Score, ...]
    global_maxima: tuple[Score, ...]

    @property
    def constant_features(self) -> tuple[bool, ...]:
        return tuple(
            minimum == maximum
            for minimum, maximum in zip(self.global_minima, self.global_maxima, strict=True)
        )

    def parameters_for(self, client_id: ClientId) -> ClientPreprocessingParameters:
        match = next((item for item in self.clients if item.client_id == client_id), None)
        if match is None:
            raise KeyError(client_id)
        return match

    def transform(self, frame: pd.DataFrame, client_id: ClientId) -> pd.DataFrame:
        """Scale by global (cross-client) min/max; output is intentionally left unclipped to [0, 1] to preserve anomaly-score geometry."""
        features = list(self.feature_columns)
        values = frame.loc[:, features].to_numpy(dtype=np.float64, copy=True)
        parameters = self.parameters_for(client_id)

        if parameters.medians is not None:
            medians = np.asarray(parameters.medians, dtype=np.float64)
            missing = ~np.isfinite(values)
            if missing.any():
                values = np.where(missing, medians, values)

        if not np.isfinite(values).all():
            raise DataIntegrityError(
                f"Non-finite values remain after preprocessing for {client_id}"
            )

        minima = np.asarray(self.global_minima, dtype=np.float64)
        span = np.asarray(self.global_maxima, dtype=np.float64) - minima

        # Constant features (span == 0) are left at 0 rather than dividing by zero.
        scaled = np.zeros_like(values)
        np.divide(values - minima, span, out=scaled, where=span.astype(bool))

        result = frame.copy()
        result.loc[:, features] = scaled
        return result


class ClientPreprocessingStatistics(BaseModel):
    model_config = Frozen

    client_id: ClientId
    feature_columns: tuple[FeatureName, ...]
    training_row_hash: Sha256
    local_minima: tuple[Score, ...]
    local_maxima: tuple[Score, ...]
    medians: tuple[Score, ...] | None = None


class TrainOnlyPreprocessing:
    def validate_training_rows(
        self,
        splits: ClientSplits,
        dataset: DatasetId,
        expected_features: PositiveCount,
        *,
        finite_rate_minimum: Probability | None = None,
    ) -> tuple[FeatureName, ...]:
        train = splits.get(DataRole.TRAIN)
        columns = model_feature_columns(train, expected_features)
        values = train.loc[:, list(columns)].to_numpy(dtype=np.float64)

        if dataset is DatasetId.NBAIOT:
            if not np.isfinite(values).all():
                raise DataIntegrityError("N-BaIoT training features must all be finite")
        elif dataset is DatasetId.DIAD:
            if finite_rate_minimum is None:
                raise DataIntegrityError(
                    "DIAD preprocessing requires a configured per-feature finite-rate minimum"
                )

            finite_rates = np.isfinite(values).mean(axis=0)
            failing = [
                columns[i] for i, rate in enumerate(finite_rates) if rate < finite_rate_minimum
            ]
            if failing:
                raise DataIntegrityError(f"DIAD_FEATURE_FINITE_RATE_FAIL: {', '.join(failing[:5])}")

        return columns

    def client_statistics(
        self,
        splits: ClientSplits,
        dataset: DatasetId,
        expected_features: PositiveCount,
        *,
        finite_rate_minimum: Probability | None = None,
    ) -> ClientPreprocessingStatistics:
        """Fits local min/max/median from this client's training rows only, for later federated aggregation."""
        columns = self.validate_training_rows(
            splits,
            dataset,
            expected_features,
            finite_rate_minimum=finite_rate_minimum,
        )

        train_frame = splits.get(DataRole.TRAIN)
        training_row_hash = hash_row_ids(train_frame[PreparedColumn.ROW_ID].astype(str).tolist())
        train_values = train_frame.loc[:, list(columns)].to_numpy(dtype=np.float64, copy=True)

        median_tuple = None
        if dataset is DatasetId.DIAD:
            finite_mask = np.isfinite(train_values)
            masked_values = np.where(finite_mask, train_values, np.nan)
            medians = np.nanmedian(masked_values, axis=0)

            if not np.isfinite(medians).all():
                raise DataIntegrityError(
                    f"DIAD imputation median is undefined for {splits.client_id}"
                )

            missing = ~finite_mask
            if missing.any():
                train_values = np.where(missing, medians, train_values)

            median_tuple = tuple(medians.tolist())

        return ClientPreprocessingStatistics(
            client_id=splits.client_id,
            feature_columns=columns,
            training_row_hash=training_row_hash,
            local_minima=tuple(np.min(train_values, axis=0).tolist()),
            local_maxima=tuple(np.max(train_values, axis=0).tolist()),
            medians=median_tuple,
        )

    def aggregate(
        self,
        statistics: tuple[ClientPreprocessingStatistics, ...],
        dataset: DatasetId,
    ) -> PreprocessingModel:
        """Elementwise min/max of clients' local extrema; this is federated communication of derived statistics, not a privacy-preserving computation."""
        if not statistics:
            raise DataIntegrityError("Cannot fit preprocessing without clients")

        ordered = tuple(sorted(statistics, key=lambda item: item.client_id))
        feature_columns = ordered[0].feature_columns

        if any(item.feature_columns != feature_columns for item in ordered):
            raise DataIntegrityError("Model feature order differs across clients")

        minima_stack = np.array([item.local_minima for item in ordered])
        maxima_stack = np.array([item.local_maxima for item in ordered])

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
            global_minima=tuple(np.min(minima_stack, axis=0).tolist()),
            global_maxima=tuple(np.max(maxima_stack, axis=0).tolist()),
        )
