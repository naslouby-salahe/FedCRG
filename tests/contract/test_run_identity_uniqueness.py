from pathlib import Path

import numpy as np
import pandas as pd

from fedcrg.application.r14 import R14FeatureSensitivityBuilder
from fedcrg.config.resolver import ExperimentConfigResolver
from fedcrg.core.enums import PolicyId
from fedcrg.core.ids import ClientId, RunId

ROOT = Path(__file__).resolve().parents[2]


def test_run_id_maps_distinct_configs_to_distinct_evidence_directories(tmp_path: Path) -> None:
    external = ExperimentConfigResolver().resolve(ROOT / "configs/experiments/external/diad.yaml")
    training = pd.DataFrame(
        {
            "row_id": [f"{index:064x}" for index in range(2000)],
            "numeric_a": np.arange(2000, dtype=float),
            "numeric_b": np.linspace(0.0, 1.0, 2000),
        }
    )
    r14 = R14FeatureSensitivityBuilder().build(
        external,
        {ClientId("diad_example0001"): training},
        tmp_path / "r14_feature_contract.json",
    ).config

    confirmatory_id = RunId.for_policy_cell(external, 11, 2000, PolicyId.FEDCRG)
    sensitivity_id = RunId.for_policy_cell(r14, 11, 2000, PolicyId.FEDCRG)

    assert external.config_hash != r14.config_hash
    assert confirmatory_id != sensitivity_id
