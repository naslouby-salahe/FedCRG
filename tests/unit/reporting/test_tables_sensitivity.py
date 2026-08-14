"""Unit tests for deterministic manuscript-table builders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fedcrg.reporting import PublicationTableBuilder
from fedcrg.types import PolicyId


def test_literature_boundary_table_is_deterministic(tmp_path: Path) -> None:
    first = PublicationTableBuilder().literature_boundary(tmp_path / "first.csv")
    second = PublicationTableBuilder().literature_boundary(tmp_path / "second.csv")
    assert first.read_bytes() == second.read_bytes()
    frame = pd.read_csv(first)
    assert set(frame["policy_id"]) == {
        PolicyId.REFERENCE_QUANTILE.value,
        PolicyId.GLOBAL_QUANTILE.value,
        PolicyId.LOCAL_QUANTILE.value,
        PolicyId.SHRINKAGE.value,
        PolicyId.THREE_SIGMA.value,
        PolicyId.FEDCRG.value,
        PolicyId.DEV_F1_SELECT.value,
        PolicyId.SUMMARY_STATISTIC_SELECT.value,
        PolicyId.SUPERVISED_F1.value,
    }
