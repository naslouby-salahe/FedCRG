import numpy as np
import pandas as pd

from fedcrg.core.ids import ClientId
from fedcrg.data.r14_feature_contract import derive_r14_feature_contract


def test_r14_keeps_stream_behavior_features_but_excludes_direct_stream_identifier() -> None:
    frame = pd.DataFrame(
        {
            "row_id": [f"{index:064x}" for index in range(100)],
            "stream": np.arange(100),
            "stream_1_count": np.arange(100, dtype=float),
            "stream_1_mean": np.linspace(0.0, 1.0, 100),
            "device_mac": np.arange(100),
            "src_port": np.arange(100),
            "numeric_behavior": np.linspace(1.0, 2.0, 100),
        }
    )
    result = derive_r14_feature_contract({ClientId("diad_example0001"): frame})
    assert "stream" not in result.features
    assert "device_mac" not in result.features
    assert "src_port" not in result.features
    assert "stream_1_count" in result.features
    assert "stream_1_mean" in result.features
    assert "numeric_behavior" in result.features
