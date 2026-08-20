"""Tests for the publication figure builders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fedcrg.reporting import (
    build_external_replication_figure,
    build_per_client_operating_points_figure,
    build_reliability_utility_frontier_figure,
)


def _assert_valid_png(path: Path) -> None:
    assert path.is_file()
    data = path.read_bytes()
    assert len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def _primary_policy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "client_id": ["nb01", "nb02", "nb01", "nb02"],
            "policy_id": ["fedcrg", "fedcrg", "reference_quantile", "reference_quantile"],
            "fpr": [0.01, 0.02, 0.03, 0.04],
            "mebe": [0.01, 0.02, 0.03, 0.04],
            "attack_balanced_macro_tpr": [0.9, 0.8, 0.7, 0.6],
        }
    )


def test_build_per_client_operating_points_figure_produces_png(tmp_path: Path) -> None:
    output = build_per_client_operating_points_figure(
        tmp_path / "figures" / "per_client.png", _primary_policy_frame()
    )
    _assert_valid_png(output)


def test_build_per_client_operating_points_figure_missing_table_raises(tmp_path: Path) -> None:
    output = tmp_path / "figures" / "per_client.png"
    frame = pd.DataFrame()
    with pytest.raises(ValueError):
        build_per_client_operating_points_figure(output, frame)


def test_build_per_client_operating_points_figure_requires_columns(tmp_path: Path) -> None:
    output = tmp_path / "figures" / "per_client.png"
    frame = pd.DataFrame({"only_column": [1, 2]})
    with pytest.raises(ValueError, match="client_id/fpr"):
        build_per_client_operating_points_figure(output, frame)


def test_build_reliability_utility_frontier_figure_produces_png(tmp_path: Path) -> None:
    output = build_reliability_utility_frontier_figure(
        tmp_path / "figures" / "reliability.png", _primary_policy_frame()
    )
    _assert_valid_png(output)


def test_build_reliability_utility_frontier_figure_requires_columns(tmp_path: Path) -> None:
    output = tmp_path / "figures" / "reliability.png"
    frame = pd.DataFrame({"policy_id": ["fedcrg"]})
    with pytest.raises(ValueError, match="MEBE/ABMacroTPR"):
        build_reliability_utility_frontier_figure(output, frame)


def test_build_external_replication_figure_produces_png(tmp_path: Path) -> None:
    frame = pd.DataFrame({"client_id": ["diad1", "diad2"], "fpr": [0.01, 0.02]})
    output = build_external_replication_figure(tmp_path / "figures" / "external.png", frame)
    _assert_valid_png(output)


def test_build_external_replication_figure_requires_columns(tmp_path: Path) -> None:
    output = tmp_path / "figures" / "external.png"
    frame = pd.DataFrame({"only_column": [1]})
    with pytest.raises(ValueError, match="client_id/fpr"):
        build_external_replication_figure(output, frame)
