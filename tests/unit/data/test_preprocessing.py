import numpy as np
import pandas as pd
import pytest

from fedcrg.core.enums import DataRole, DatasetId
from fedcrg.core.exceptions import DataIntegrityError
from fedcrg.data.models import ClientSplits
from fedcrg.data.preprocessing import FederatedPreprocessor


def _splits(values: np.ndarray) -> ClientSplits:
    frame = pd.DataFrame(values, columns=["f1", "f2"])
    frame["row_id"] = [f"r{i}" for i in range(len(frame))]
    return ClientSplits("c1", {DataRole.TRAIN: frame})


def test_nbaiot_preprocessing_uses_global_train_extrema_without_clipping() -> None:
    first = _splits(np.array([[0.0, 5.0], [1.0, 5.0]]))
    second_frame = pd.DataFrame({"f1": [2.0, 3.0], "f2": [5.0, 5.0], "row_id": ["x", "y"]})
    second = ClientSplits("c2", {DataRole.TRAIN: second_frame})
    model = FederatedPreprocessor().fit({"c1": first, "c2": second}, DatasetId.NBAIOT, expected_features=2)
    test = pd.DataFrame({"f1": [4.0], "f2": [5.0], "row_id": ["z"]})
    transformed = model.transform(test, "c1")
    assert transformed.loc[0, "f1"] > 1.0
    assert transformed.loc[0, "f2"] == 0.0
    assert model.constant_features == (False, True)


def test_diad_preprocessing_imputes_from_client_train_median() -> None:
    frame = pd.DataFrame({"f1": np.arange(100, dtype=float), "f2": np.arange(100, dtype=float) * 2.0, "row_id": [f"r{i}" for i in range(100)]})
    frame.loc[0, "f1"] = np.nan
    splits = ClientSplits("c1", {DataRole.TRAIN: frame})
    model = FederatedPreprocessor().fit({"c1": splits}, DatasetId.DIAD, expected_features=2)
    assert model.client_imputers["c1"].medians == (50.0, 99.0)
    transformed = model.transform(frame, "c1")
    assert np.isfinite(transformed[["f1", "f2"]].to_numpy()).all()


def test_diad_training_finite_rate_is_enforced() -> None:
    values = np.ones((100, 2))
    values[:2, 0] = np.nan
    with pytest.raises(DataIntegrityError, match="DIAD_FEATURE_FINITE_RATE_FAIL"):
        FederatedPreprocessor().validate_training_rows(_splits(values), DatasetId.DIAD, expected_features=2)
