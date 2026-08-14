"""Unit tests for the federation-wide reference threshold estimator."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import TypeAdapter

from fedcrg.thresholding.readiness import (
    build_reference_threshold,
    reference_rank,
)
from fedcrg.types import ClientId

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)


def test_reference_rank_is_locked() -> None:
    assert reference_rank(100, 0.01) == 100
    assert reference_rank(736, 0.01) == 730


def test_estimator_pools_equal_counts_across_clients() -> None:
    scores = {
        _CLIENT_ID_ADAPTER.validate_python("client-a"): np.zeros(2),
        _CLIENT_ID_ADAPTER.validate_python("client-b"): np.ones(2),
    }
    reference = build_reference_threshold(scores, 0.01)
    assert reference.sample_count == 4
    assert reference.client_count == 2
    assert reference.samples_per_client == 2
    assert reference.rank == 4
    assert reference.value == 1.0


def test_estimator_rejects_unequal_contributions() -> None:
    scores = {
        _CLIENT_ID_ADAPTER.validate_python("client-a"): np.zeros(2),
        _CLIENT_ID_ADAPTER.validate_python("client-b"): np.ones(3),
    }
    with pytest.raises(ValueError):
        build_reference_threshold(scores, 0.01)


def test_estimator_rejects_empty_pool() -> None:
    with pytest.raises(ValueError):
        build_reference_threshold({}, 0.01)
