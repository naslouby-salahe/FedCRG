import numpy as np

from fedcrg.domain.identifiers import ClientId
from fedcrg.protocol.reference import ReferenceThresholdEstimator, reference_rank


def test_reference_rank_exact_primary_value() -> None:
    assert reference_rank(4500, 0.01) == 4456


def test_reference_estimator_requires_equal_client_contribution() -> None:
    estimator = ReferenceThresholdEstimator()
    with np.testing.assert_raises(ValueError):
        estimator.estimate({ClientId("a"): np.zeros(2), ClientId("b"): np.zeros(3)}, 0.01)
