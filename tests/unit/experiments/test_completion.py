from __future__ import annotations

from pathlib import Path

import pytest

from fedcrg.config import Study
from fedcrg.experiments.analyses import RunSyntheticExperiments
from fedcrg.types import ExperimentId


def test_trial_ledger_rejects_shortfall() -> None:
    spec = Study.load().spec(ExperimentId.TARGET_FPR_SYNTHETIC)
    assert spec.workload.monte_carlo_trials > 0
    with pytest.raises(RuntimeError, match="trial ledger mismatch"):
        RunSyntheticExperiments.write(Path("out.json"), spec, cells=(), actual_trials=0)


def test_cell_ledger_rejects_shortfall() -> None:
    spec = Study.load().spec(ExperimentId.MISMATCH_POWER)
    assert spec.workload.exact_cells == 45
    with pytest.raises(RuntimeError, match="cell ledger mismatch"):
        RunSyntheticExperiments.write(Path("out.json"), spec, cells=(), actual_trials=0)
