import numpy as np

from fedcrg.domain.enums import MismatchOutcome
from fedcrg.domain.values import OperatingBand
from fedcrg.protocol.mismatch import ReferenceMismatchEvaluator, minimum_bidirectional_sample_count


def _scores(n: int, x: int) -> np.ndarray:
    return np.array([1.0] * x + [-1.0] * (n - x), dtype=np.float64)


def test_primary_minimum_sample_count() -> None:
    assert minimum_bidirectional_sample_count(0.005, 0.95) == 736


def test_exact_mismatch_cutoffs() -> None:
    evaluator = ReferenceMismatchEvaluator()
    band = OperatingBand(0.005, 0.015)
    cases = [
        (736, 0, MismatchOutcome.LOW),
        (736, 1, MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        (1500, 2, MismatchOutcome.LOW),
        (1500, 3, MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        (1500, 32, MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        (1500, 33, MismatchOutcome.HIGH),
        (3000, 7, MismatchOutcome.LOW),
        (3000, 8, MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        (3000, 58, MismatchOutcome.NO_MATERIAL_DIFFERENCE),
        (3000, 59, MismatchOutcome.HIGH),
    ]
    for n, x, expected in cases:
        assert evaluator.evaluate(_scores(n, x), 0.0, band, 0.95).outcome is expected


def test_insufficient_evidence_is_explicit() -> None:
    result = ReferenceMismatchEvaluator().evaluate(
        np.zeros(735), 1.0, OperatingBand(0.005, 0.015), 0.95
    )
    assert result.outcome is MismatchOutcome.INSUFFICIENT_EVIDENCE
    assert result.sample_count < (result.minimum_sample_count or 0)
    assert result.interval is not None
