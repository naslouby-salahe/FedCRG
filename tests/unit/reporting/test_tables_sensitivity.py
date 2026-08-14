from __future__ import annotations

from pathlib import Path

import pandas as pd

from fedcrg.reporting import PublicationTableBuilder
from tests._fixtures import primary_experiment_config


def test_protocol_constants_table_is_deterministic(tmp_path: Path) -> None:
    config = primary_experiment_config(tmp_path / "outputs")
    first = PublicationTableBuilder().protocol_constants(config, tmp_path / "first.csv")
    second = PublicationTableBuilder().protocol_constants(config, tmp_path / "second.csv")
    assert first.read_bytes() == second.read_bytes()
    frame = pd.read_csv(first)
    assert not frame.empty
    assert "constant" in frame.columns
