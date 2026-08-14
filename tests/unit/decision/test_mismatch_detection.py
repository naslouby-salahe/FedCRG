from __future__ import annotations

import numpy as np
import pytest

from fedcrg.thresholding.readiness import (
    ReferenceMismatchEvaluator,
    minimum_bidirectional_sample_count,
)
from fedcrg.types import MismatchOutcome
from tests._fixtures import primary_protocol


def test_minimum_bidirectional_sample_count_is_locked() -> None:
    protocol = primary_protocol()
    minimum = minimum_bidirectional_sample_count(protocol.band.lower, protocol.mismatch_confidence)
    assert minimum == 736


def test_evaluator_detects_high_mismatch() -> None:
    protocol = primary_protocol()
    band = protocol.band
    evaluator = ReferenceMismatchEvaluator()
    result = evaluator.evaluate(
        scores=np.linspace(0.75, 0.95, 736),
        reference_threshold=0.75,
        band=band,
        confidence=protocol.mismatch_confidence,
    )
    assert result.outcome is MismatchOutcome.HIGH
    assert result.exceedance_count > 0
    assert result.sample_count == 736
    assert result.estimated_fpr > 0.0


def test_evaluator_returns_insufficient_evidence_below_minimum() -> None:
    protocol = primary_protocol()
    evaluator = ReferenceMismatchEvaluator()
    result = evaluator.evaluate(
        scores=np.linspace(0.8, 0.9, 100),
        reference_threshold=0.75,
        band=protocol.band,
        confidence=protocol.mismatch_confidence,
    )
    assert result.outcome is MismatchOutcome.INSUFFICIENT_EVIDENCE
    assert result.exceedance_count == 100


def test_evaluator_rejects_empty_scores() -> None:
    protocol = primary_protocol()
    evaluator = ReferenceMismatchEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(
            scores=np.array([]),
            reference_threshold=0.75,
            band=protocol.band,
            confidence=protocol.mismatch_confidence,
        )
