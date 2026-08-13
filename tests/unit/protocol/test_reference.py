import numpy as np

from fedcrg.protocol.reference import ReferenceThresholdEstimator, reference_rank


def test_reference_rank_exact_primary_value() -> None:
    assert reference_rank(4500, 0.01) == 4456


def test_reference_estimator_requires_equal_client_contribution() -> None:
    estimator = ReferenceThresholdEstimator()
    with np.testing.assert_raises(ValueError):
        estimator.estimate({"a": np.zeros(2), "b": np.zeros(3)}, 0.01)
