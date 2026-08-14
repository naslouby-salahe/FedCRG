"""Unit tests for calibration-readiness evaluation and plan building."""

from __future__ import annotations

import numpy as np
import pytest

from fedcrg.thresholding.readiness import (
    CalibrationReadiness,
    CalibrationReadinessEvaluator,
    ReadinessPlan,
    ReadinessPlanBuilder,
)
from fedcrg.types import CalibrationReadinessState, OperatingBand
from tests._fixtures import primary_protocol


def test_readiness_plan_matches_normative_shape() -> None:
    protocol = primary_protocol()
    plan = ReadinessPlanBuilder().build(2000, protocol.band, protocol.readiness_assurance)
    assert plan.sample_count == 2000
    assert plan.state is CalibrationReadinessState.READY
    assert plan.coverage_probability >= protocol.readiness_assurance - 1e-12
    assert 0.0 <= plan.rank <= 2000


def test_readiness_plan_rejects_implausible_coverage() -> None:
    protocol = primary_protocol()
    builder = ReadinessPlanBuilder()
    with pytest.raises(ValueError):
        builder.build(2000, protocol.band, 1.5)
    with pytest.raises(ValueError):
        builder.build(0, protocol.band, protocol.readiness_assurance)


def test_readiness_plan_for_small_sample_is_not_ready() -> None:
    protocol = primary_protocol()
    plan = ReadinessPlanBuilder().build(100, protocol.band, protocol.readiness_assurance)
    assert plan.state is CalibrationReadinessState.NOT_READY
    assert plan.coverage_probability < protocol.readiness_assurance


def test_evaluator_can_fail_a_plan() -> None:
    protocol = primary_protocol()
    plan = ReadinessPlanBuilder().build(100, protocol.band, protocol.readiness_assurance)
    evaluator = CalibrationReadinessEvaluator()
    low = np.linspace(0.0, 0.4, 100)
    result = evaluator.evaluate(low, plan)
    assert isinstance(result, CalibrationReadiness)
    assert result.plan is plan
    assert result.threshold is None
    assert result.tie_count == 1
    assert result.diagnostics.unique_score_fraction == 1.0


def test_evaluator_rejects_size_mismatch() -> None:
    protocol = primary_protocol()
    plan = ReadinessPlanBuilder().build(2000, protocol.band, protocol.readiness_assurance)
    evaluator = CalibrationReadinessEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(np.linspace(0.0, 1.0, 100), plan)
    with pytest.raises(ValueError):
        evaluator.evaluate(np.array([]), plan)


def test_plan_roundtrip_via_model() -> None:
    protocol = primary_protocol()
    plan = ReadinessPlanBuilder().build(2000, protocol.band, protocol.readiness_assurance)
    decoded = ReadinessPlan.model_validate_json(plan.model_dump_json())
    assert decoded == plan
    assert decoded.state is CalibrationReadinessState.READY
    assert isinstance(decoded.band, OperatingBand)
