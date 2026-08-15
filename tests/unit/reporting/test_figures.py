"""Tests for the publication figure builders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fedcrg.paths import ResultsBundleLayout
from fedcrg.reporting import (
    PublicationTableFilename,
    build_assumption_stress_figure,
    build_decision_architecture_figure,
    build_external_replication_figure,
    build_mismatch_power_map_figure,
    build_mismatch_power_map_from_catalogue,
    build_per_client_operating_points_figure,
    build_phase_transition_figure,
    build_phase_transition_from_catalogue,
    build_readiness_frontier_figure,
    build_readiness_frontier_from_catalogue,
    build_reliability_utility_frontier_figure,
)


def _assert_valid_png(path: Path) -> None:
    assert path.is_file()
    data = path.read_bytes()
    assert len(data) > 0
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_decision_architecture_figure_produces_png(tmp_path: Path) -> None:
    output = build_decision_architecture_figure(tmp_path / "figures" / "arch.png")
    _assert_valid_png(output)


def test_build_readiness_frontier_figure_marks_minimum_when_reached(tmp_path: Path) -> None:
    from fedcrg.config import OperatingBand

    band = OperatingBand(lower=0.001, upper=0.02)
    output = build_readiness_frontier_figure(
        tmp_path / "frontier.png",
        (500, 1000, 1500, 2000),
        band,
        0.95,
    )
    _assert_valid_png(output)


def test_build_readiness_frontier_figure_without_reaching_minimum(tmp_path: Path) -> None:
    from fedcrg.config import OperatingBand

    band = OperatingBand(lower=1e-9, upper=2e-9)
    output = build_readiness_frontier_figure(
        tmp_path / "frontier_unreached.png",
        (10, 20),
        band,
        0.999999,
    )
    _assert_valid_png(output)


def test_build_mismatch_power_map_figure_produces_png(tmp_path: Path) -> None:
    from fedcrg.config import OperatingBand

    band = OperatingBand(lower=0.005, upper=0.015)
    output = build_mismatch_power_map_figure(
        tmp_path / "power_map.png",
        (50, 100),
        (0.005, 0.01, 0.02),
        band,
        0.95,
    )
    _assert_valid_png(output)


def test_build_phase_transition_figure_produces_png(tmp_path: Path) -> None:
    from fedcrg.config import OperatingBand

    band = OperatingBand(lower=0.001, upper=0.02)
    output = build_phase_transition_figure(
        tmp_path / "phase.png",
        (500, 1000, 1500),
        (100, 200, 300),
        band,
        0.95,
        0.95,
    )
    _assert_valid_png(output)


def test_build_readiness_frontier_from_catalogue_produces_png(tmp_path: Path) -> None:
    output = build_readiness_frontier_from_catalogue(tmp_path / "catalogue_frontier.png")
    _assert_valid_png(output)


def test_build_mismatch_power_map_from_catalogue_produces_png(tmp_path: Path) -> None:
    output = build_mismatch_power_map_from_catalogue(tmp_path / "catalogue_power.png")
    _assert_valid_png(output)


def test_build_phase_transition_from_catalogue_produces_png(tmp_path: Path) -> None:
    output = build_phase_transition_from_catalogue(tmp_path / "catalogue_phase.png")
    _assert_valid_png(output)


def _write_primary_policy_results_table(results_root: Path) -> None:
    layout = ResultsBundleLayout(results_root)
    layout.tables.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "client_id": ["nb01", "nb02", "nb01", "nb02"],
            "policy_id": ["fedcrg", "fedcrg", "reference_quantile", "reference_quantile"],
            "fpr": [0.01, 0.02, 0.03, 0.04],
            "mebe": [0.01, 0.02, 0.03, 0.04],
            "attack_balanced_macro_tpr": [0.9, 0.8, 0.7, 0.6],
        }
    )
    frame.to_csv(layout.tables / PublicationTableFilename.PRIMARY_POLICY_RESULTS, index=False)


def test_build_per_client_operating_points_figure_produces_png(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_primary_policy_results_table(results_root)
    output = build_per_client_operating_points_figure(
        tmp_path / "figures" / "per_client.png", results_root
    )
    _assert_valid_png(output)


def test_build_per_client_operating_points_figure_missing_table_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_per_client_operating_points_figure(
            tmp_path / "figures" / "per_client.png", tmp_path / "results"
        )


def test_build_per_client_operating_points_figure_requires_columns(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    layout = ResultsBundleLayout(results_root)
    layout.tables.mkdir(parents=True)
    pd.DataFrame({"only_column": [1, 2]}).to_csv(
        layout.tables / PublicationTableFilename.PRIMARY_POLICY_RESULTS, index=False
    )
    with pytest.raises(ValueError, match="client_id/fpr"):
        build_per_client_operating_points_figure(
            tmp_path / "figures" / "per_client.png", results_root
        )


def test_build_reliability_utility_frontier_figure_produces_png(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_primary_policy_results_table(results_root)
    output = build_reliability_utility_frontier_figure(
        tmp_path / "figures" / "reliability.png", results_root
    )
    _assert_valid_png(output)


def test_build_reliability_utility_frontier_figure_requires_columns(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    layout = ResultsBundleLayout(results_root)
    layout.tables.mkdir(parents=True)
    pd.DataFrame({"policy_id": ["fedcrg"]}).to_csv(
        layout.tables / PublicationTableFilename.PRIMARY_POLICY_RESULTS, index=False
    )
    with pytest.raises(ValueError, match="MEBE/ABMacroTPR"):
        build_reliability_utility_frontier_figure(
            tmp_path / "figures" / "reliability.png", results_root
        )


def test_build_assumption_stress_figure_produces_png(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    layout = ResultsBundleLayout(results_root)
    layout.statistics.mkdir(parents=True)
    (layout.statistics / "cell_a.json").write_text(json.dumps({"coverage": 0.94}), encoding="utf-8")
    (layout.statistics / "cell_b.json").write_text(json.dumps({"coverage": 0.96}), encoding="utf-8")
    output = build_assumption_stress_figure(tmp_path / "figures" / "stress.png", results_root)
    _assert_valid_png(output)


def test_build_assumption_stress_figure_missing_statistics_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_assumption_stress_figure(tmp_path / "figures" / "stress.png", tmp_path / "results")


def test_build_assumption_stress_figure_no_coverage_entries_raises(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    layout = ResultsBundleLayout(results_root)
    layout.statistics.mkdir(parents=True)
    (layout.statistics / "cell_a.json").write_text(json.dumps({"other": 1}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="No coverage statistics"):
        build_assumption_stress_figure(tmp_path / "figures" / "stress.png", results_root)


def test_build_external_replication_figure_produces_png(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    layout = ResultsBundleLayout(results_root)
    layout.tables.mkdir(parents=True)
    frame = pd.DataFrame({"client_id": ["diad1", "diad2"], "fpr": [0.01, 0.02]})
    frame.to_csv(layout.tables / PublicationTableFilename.EXTERNAL_REPLICATION, index=False)
    output = build_external_replication_figure(tmp_path / "figures" / "external.png", results_root)
    _assert_valid_png(output)


def test_build_external_replication_figure_requires_columns(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    layout = ResultsBundleLayout(results_root)
    layout.tables.mkdir(parents=True)
    pd.DataFrame({"only_column": [1]}).to_csv(
        layout.tables / PublicationTableFilename.EXTERNAL_REPLICATION, index=False
    )
    with pytest.raises(ValueError, match="client_id/fpr"):
        build_external_replication_figure(tmp_path / "figures" / "external.png", results_root)
