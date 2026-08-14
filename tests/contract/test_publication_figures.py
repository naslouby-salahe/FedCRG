"""Contract: publication figures are deterministic and honest.

The formula-driven figures (readiness frontier, mismatch power map, phase
transition) must render byte-identically for identical inputs; the
results-driven figures must refuse to fabricate output when the required
bundle tables are absent.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest

from fedcrg.reporting import (
    build_mismatch_power_map_figure,
    build_phase_transition_figure,
    build_readiness_frontier_figure,
    build_per_client_operating_points_figure,
    build_reliability_utility_frontier_figure,
    build_assumption_stress_figure,
    build_external_replication_figure,
)
from fedcrg.types import OperatingBand


def test_formula_figures_are_deterministic(tmp_path: Path) -> None:
    band = OperatingBand(lower=0.005, upper=0.015)
    counts = (500, 1000, 1400, 1415, 1416, 1500, 2000)
    gate_counts = (736, 1000, 1500, 2000, 3000)
    true_fprs = (0.0025, 0.005, 0.01, 0.015, 0.03)

    builders: tuple[Callable[[Path], Path], ...] = (
        partial(build_readiness_frontier_figure, sample_counts=counts, band=band, assurance=0.95),
        partial(
            build_mismatch_power_map_figure,
            sample_counts=gate_counts,
            true_fprs=true_fprs,
            band=band,
            confidence=0.95,
        ),
        partial(
            build_phase_transition_figure,
            calibration_counts=counts,
            gate_counts=gate_counts,
            band=band,
            assurance=0.95,
        ),
    )
    for index, builder in enumerate(builders):
        first = builder(tmp_path / f"first_{index}.png")
        second = builder(tmp_path / f"second_{index}.png")
        assert first.read_bytes() == second.read_bytes()
        assert first.stat().st_size > 1000


def test_results_driven_figures_never_fabricate(tmp_path: Path) -> None:
    output = tmp_path / "figure.png"
    empty_results = tmp_path / "results"
    with pytest.raises(FileNotFoundError):
        build_per_client_operating_points_figure(output, empty_results)
    with pytest.raises(FileNotFoundError):
        build_reliability_utility_frontier_figure(output, empty_results)
    with pytest.raises(FileNotFoundError):
        build_assumption_stress_figure(output, empty_results)
    with pytest.raises(FileNotFoundError):
        build_external_replication_figure(output, empty_results)
    assert not output.exists()
