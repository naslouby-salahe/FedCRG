"""Unit tests for train-only preprocessing statistics and transform."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from fedcrg.data.preprocessing import TrainOnlyPreprocessing
from fedcrg.data.splits import ClientSplits, RoleFrame
from fedcrg.types import ClientId, DataIntegrityError, DataRole, DatasetId

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
C1 = _CLIENT_ID_ADAPTER.validate_python("c1")
C2 = _CLIENT_ID_ADAPTER.validate_python("c2")


def _splits(values: np.ndarray) -> ClientSplits:
    frame = pd.DataFrame(values, columns=["f1", "f2"])
    frame["row_id"] = [f"r{i}" for i in range(len(frame))]
    return ClientSplits(client_id=C1, roles=(RoleFrame(role=DataRole.TRAIN, frame=frame),))


def test_nbaiot_preprocessing_uses_global_train_extrema_without_clipping() -> None:
    first = _splits(np.array([[0.0, 5.0], [1.0, 5.0]]))
    second_frame = pd.DataFrame({"f1": [2.0, 3.0], "f2": [5.0, 5.0], "row_id": ["x", "y"]})
    second = ClientSplits(client_id=C2, roles=(RoleFrame(role=DataRole.TRAIN, frame=second_frame),))
    preprocessor = TrainOnlyPreprocessing()
    statistics = (
        preprocessor.client_statistics(first, DatasetId.NBAIOT, expected_features=2),
        preprocessor.client_statistics(second, DatasetId.NBAIOT, expected_features=2),
    )
    model = preprocessor.aggregate(statistics, DatasetId.NBAIOT)
    test = pd.DataFrame({"f1": [4.0], "f2": [5.0], "row_id": ["z"]})
    transformed = model.transform(test, C1)
    assert float(transformed["f1"].iloc[0]) > 1.0
    assert float(transformed["f2"].iloc[0]) == 0.0
    assert model.constant_features == (False, True)


def test_diad_preprocessing_imputes_from_client_train_median() -> None:
    frame = pd.DataFrame(
        {
            "f1": np.arange(100, dtype=float),
            "f2": np.arange(100, dtype=float) * 2.0,
            "row_id": [f"r{i}" for i in range(100)],
        }
    )
    frame.loc[0, "f1"] = np.nan
    splits = ClientSplits(client_id=C1, roles=(RoleFrame(role=DataRole.TRAIN, frame=frame),))
    preprocessor = TrainOnlyPreprocessing()
    statistics = (
        preprocessor.client_statistics(
            splits, DatasetId.DIAD, expected_features=2, finite_rate_minimum=0.99
        ),
    )
    model = preprocessor.aggregate(statistics, DatasetId.DIAD)
    assert model.parameters_for(C1).medians == (50.0, 99.0)
    transformed = model.transform(frame, C1)
    assert np.isfinite(transformed[["f1", "f2"]].to_numpy()).all()


def test_diad_training_finite_rate_is_enforced() -> None:
    values = np.ones((100, 2))
    values[:2, 0] = np.nan
    with pytest.raises(DataIntegrityError, match="DIAD_FEATURE_FINITE_RATE_FAIL"):
        TrainOnlyPreprocessing().validate_training_rows(
            _splits(values),
            DatasetId.DIAD,
            expected_features=2,
            finite_rate_minimum=0.99,
        )


def test_aggregate_rejects_feature_order_drift() -> None:
    preprocessor = TrainOnlyPreprocessing()
    first = _splits(np.array([[0.0, 5.0], [1.0, 5.0]]))
    second_frame = pd.DataFrame({"f2": [5.0, 5.0], "f1": [2.0, 3.0], "row_id": ["x", "y"]})
    second = ClientSplits(client_id=C2, roles=(RoleFrame(role=DataRole.TRAIN, frame=second_frame),))
    statistics = (
        preprocessor.client_statistics(first, DatasetId.NBAIOT, expected_features=2),
        preprocessor.client_statistics(second, DatasetId.NBAIOT, expected_features=2),
    )
    with pytest.raises(DataIntegrityError):
        preprocessor.aggregate(statistics, DatasetId.NBAIOT)
