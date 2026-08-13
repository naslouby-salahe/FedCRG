from __future__ import annotations

import numpy as np
import pytest

from fedcrg.core.enums import CalibrationReadinessState, MismatchOutcome
from fedcrg.core.types import OperatingBand
from fedcrg.protocol.mismatch import ReferenceMismatchEvaluator, minimum_bidirectional_sample_count
from fedcrg.protocol.readiness import ReadinessPlanBuilder
from fedcrg.protocol.reference import reference_rank

BAND = OperatingBand(0.005, 0.015)


@pytest.mark.parametrize(
    ("sample_count", "expected_state", "expected_rank", "expected_probability"),
    [
        (1415, CalibrationReadinessState.NOT_READY, 1403, 0.9499884311),
        (1416, CalibrationReadinessState.READY, 1404, 0.9500045311),
        (1500, CalibrationReadinessState.READY, 1487, 0.9573928914),
        (2000, CalibrationReadinessState.READY, 1982, 0.9805279151),
    ],
)
def test_readiness_numerical_ledger(
    sample_count: int,
    expected_state: CalibrationReadinessState,
    expected_rank: int,
    expected_probability: float,
) -> None:
    plan = ReadinessPlanBuilder().build(sample_count, BAND, 0.95)
    assert plan.state is expected_state
    assert plan.rank == expected_rank
    assert plan.coverage_probability == pytest.approx(expected_probability, abs=1e-10)


@pytest.mark.parametrize(
    ("sample_count", "exceedances", "expected"),
    [
        (736, 0, MismatchOutcome.LOW),
        (736, 1, MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        (1000, 0, MismatchOutcome.LOW),
        (1000, 1, MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        (1000, 23, MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        (1000, 24, MismatchOutcome.HIGH),
        (1500, 2, MismatchOutcome.LOW),
        (1500, 3, MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        (1500, 32, MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        (1500, 33, MismatchOutcome.HIGH),
        (3000, 7, MismatchOutcome.LOW),
        (3000, 8, MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        (3000, 58, MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        (3000, 59, MismatchOutcome.HIGH),
    ],
)
def test_mismatch_cutoff_ledger(
    sample_count: int, exceedances: int, expected: MismatchOutcome
) -> None:
    scores = np.concatenate(
        (
            np.ones(exceedances, dtype=np.float64),
            np.zeros(sample_count - exceedances, dtype=np.float64),
        )
    )
    result = ReferenceMismatchEvaluator().evaluate(scores, 0.5, BAND, 0.95)
    assert result.outcome is expected


def test_primary_minimum_mismatch_sample_count() -> None:
    assert minimum_bidirectional_sample_count(0.005, 0.95) == 736
    assert minimum_bidirectional_sample_count(0.0, 0.95) is None


def test_reference_rank_primary_cell() -> None:
    assert reference_rank(4500, 0.01) == 4456
