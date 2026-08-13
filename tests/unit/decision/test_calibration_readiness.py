import numpy as np
from fedcrg.domain.enums import CalibrationReadinessState
from fedcrg.domain.values import OperatingBand
from fedcrg.decision.calibration_readiness import (
    CalibrationReadinessEvaluator,
    ReadinessPlanBuilder,
)


def test_primary_readiness_exact_values() -> None:
    planner = ReadinessPlanBuilder()
    band = OperatingBand(0.005, 0.015)
    cases = [
        (1415, 1403, 0.9499884311, CalibrationReadinessState.NOT_READY),
        (1416, 1404, 0.9500045311, CalibrationReadinessState.READY),
        (1500, 1487, 0.9573928914, CalibrationReadinessState.READY),
        (2000, 1982, 0.9805279151, CalibrationReadinessState.READY),
    ]
    for n, rank, probability, state in cases:
        plan = planner.build(n, band, 0.95)
        assert plan.rank == rank
        assert abs(plan.coverage_probability - probability) <= 1e-10
        assert plan.state is state


def test_readiness_plan_is_score_value_independent() -> None:
    planner = ReadinessPlanBuilder()
    evaluator = CalibrationReadinessEvaluator()
    band = OperatingBand(0.005, 0.015)
    plan = planner.build(1500, band, 0.95)
    low = evaluator.evaluate(np.arange(1500, dtype=float), plan)
    high = evaluator.evaluate(np.arange(1500, dtype=float) + 10000.0, plan)
    assert low.plan == high.plan
    assert low.threshold is not None
    assert high.threshold is not None
    assert high.threshold - low.threshold == 10000.0
