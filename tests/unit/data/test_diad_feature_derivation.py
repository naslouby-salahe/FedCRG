from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from fedcrg.config import DiadFeatureDerivation, Study
from fedcrg.data.diad import ClientTrainingFrame, derive_numeric_safe_features
from fedcrg.types import ClientId, DatasetId

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)


def _derivation() -> DiadFeatureDerivation:
    derivation = Study.load().datasets.contract(DatasetId.DIAD).diad_feature_derivation
    assert derivation is not None
    return derivation


_DERIVATION = _derivation()


def _stream_frame(rows: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"{i:064x}" for i in range(rows)],
            "stream": np.zeros(rows),
            "stream_1_count": np.arange(rows) % 7,
            "stream_1_mean": np.linspace(0.0, 1.0, rows),
            "device_mac": "aa:bb:cc:dd:ee:ff",
            "src_port": 443,
            "numeric_behavior": np.linspace(0.5, 2.5, rows),
        }
    )


def _training_frames() -> tuple[ClientTrainingFrame, ...]:
    client = _CLIENT_ID_ADAPTER.validate_python("client-a")
    return (ClientTrainingFrame(client_id=client, frame=_stream_frame()),)


def test_derive_numeric_safe_features_keeps_only_numeric_stream_features() -> None:
    result = derive_numeric_safe_features(_training_frames(), _DERIVATION)
    assert result.features == ("numeric_behavior", "stream_1_count", "stream_1_mean")
    assert result.dimension == 3
    assert result.encoder_hidden_dims == (2, 1, 1, 1)
    assert len(result.training_row_hashes) == 1


def test_derive_numeric_safe_features_is_deterministic() -> None:
    first = derive_numeric_safe_features(_training_frames(), _DERIVATION)
    second = derive_numeric_safe_features(_training_frames(), _DERIVATION)
    assert first.features == second.features
    assert first.architecture == second.architecture
    assert first.training_row_hashes == second.training_row_hashes


def test_derive_numeric_safe_features_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        derive_numeric_safe_features((), _DERIVATION)
