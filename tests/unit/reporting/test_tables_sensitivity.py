import json
from pathlib import Path

import pandas as pd

from fedcrg.reporting.tables import PublicationTableBuilder


def test_sensitivity_reads_cells_directory_not_a_single_results_file(tmp_path: Path) -> None:
    """Each real-score sensitivity writes one SensitivityEnvelope/MultiplicityEnvelope per
    model seed under experiments/{experiment_id}/cells/*.json, never a single
    experiments/{experiment_id}/results.json."""
    cells_root = tmp_path / "readiness_sample_size" / "cells"
    cells_root.mkdir(parents=True)
    (cells_root / "11.json").write_text(
        json.dumps(
            {
                "experiment_id": "readiness_sample_size",
                "model_seed": 11,
                "calibration_seed": 1000,
                "cells": [
                    {"settings": [{"axis": "calibration_n", "value": 30}], "config_hash": "x"}
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "table.csv"
    result_path = PublicationTableBuilder().sensitivity(tmp_path, output)
    frame = pd.read_csv(result_path)
    assert len(frame) == 1
    assert frame.iloc[0]["experiment_id"] == "readiness_sample_size"
    assert frame.iloc[0]["model_seed"] == 11
    assert frame.iloc[0]["calibration_seed"] == 1000
