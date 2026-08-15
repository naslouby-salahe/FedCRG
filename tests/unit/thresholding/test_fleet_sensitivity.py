from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from fedcrg.thresholding.readiness import (
    bonferroni_fleet_sensitivity,
    clopper_pearson_interval,
    familywise_readiness_assurance,
    holm_directional_fleet_sensitivity,
    minimum_bidirectional_sample_count,
)
from fedcrg.types import BinomialCounts, ClientId, MismatchOutcome, OperatingBand

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_BAND = OperatingBand(lower=0.005, upper=0.015)


def _client(name: str) -> ClientId:
    return _CLIENT_ID_ADAPTER.validate_python(name)


def test_familywise_readiness_assurance_matches_bonferroni_formula() -> None:
    assert familywise_readiness_assurance(4, 0.05) == pytest.approx(1.0 - 0.05 / 4)


def test_familywise_readiness_assurance_rejects_nonpositive_client_count() -> None:
    with pytest.raises(ValueError):
        familywise_readiness_assurance(0, 0.05)


def test_familywise_readiness_assurance_rejects_out_of_range_alpha() -> None:
    with pytest.raises(ValueError):
        familywise_readiness_assurance(4, 0.0)
    with pytest.raises(ValueError):
        familywise_readiness_assurance(4, 1.5)


def test_clopper_pearson_interval_rejects_out_of_range_confidence() -> None:
    counts = BinomialCounts(x=1, n=10)
    with pytest.raises(ValueError):
        clopper_pearson_interval(counts, 0.0)
    with pytest.raises(ValueError):
        clopper_pearson_interval(counts, 1.0)


def test_clopper_pearson_interval_pins_boundaries_at_extreme_counts() -> None:
    lower_interval = clopper_pearson_interval(BinomialCounts(x=0, n=50), 0.95)
    assert lower_interval.lower == 0.0
    upper_interval = clopper_pearson_interval(BinomialCounts(x=50, n=50), 0.95)
    assert upper_interval.upper == 1.0


def test_minimum_bidirectional_sample_count_rejects_out_of_range_lower_band() -> None:
    with pytest.raises(ValueError):
        minimum_bidirectional_sample_count(1.0, 0.95)
    with pytest.raises(ValueError):
        minimum_bidirectional_sample_count(-0.1, 0.95)


def test_minimum_bidirectional_sample_count_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        minimum_bidirectional_sample_count(0.05, 0.0)
    with pytest.raises(ValueError):
        minimum_bidirectional_sample_count(0.05, 1.0)


def test_minimum_bidirectional_sample_count_is_none_for_zero_lower_band() -> None:
    assert minimum_bidirectional_sample_count(0.0, 0.95) is None


def _assert_minimal_bidirectional_sample_count(
    lower_band: float, confidence: float, estimate: int
) -> None:
    tail = (1.0 - confidence) / 2.0
    assert 1.0 - tail ** (1.0 / estimate) < lower_band
    if estimate > 1:
        assert 1.0 - tail ** (1.0 / (estimate - 1)) >= lower_band


def test_minimum_bidirectional_sample_count_corrects_upward_after_initial_estimate() -> None:
    estimate = minimum_bidirectional_sample_count(0.75, 0.5)
    assert estimate == 2
    _assert_minimal_bidirectional_sample_count(0.75, 0.5, estimate)


def test_minimum_bidirectional_sample_count_corrects_downward_after_initial_estimate() -> None:
    estimate = minimum_bidirectional_sample_count(0.299, 0.017197999999999825)
    assert estimate == 2
    _assert_minimal_bidirectional_sample_count(0.299, 0.017197999999999825, estimate)


def test_minimum_bidirectional_sample_count_satisfies_minimality_property() -> None:
    for lower_band, confidence in (
        (0.005, 0.95),
        (0.05, 0.99),
        (0.2, 0.9),
        (0.4, 0.999),
        (0.001, 0.5),
    ):
        estimate = minimum_bidirectional_sample_count(lower_band, confidence)
        assert estimate is not None
        _assert_minimal_bidirectional_sample_count(lower_band, confidence, estimate)


def test_bonferroni_fleet_sensitivity_is_empty_for_no_clients() -> None:
    assert bonferroni_fleet_sensitivity({}, _BAND, familywise_alpha=0.05) == ()


def test_bonferroni_fleet_sensitivity_classifies_low_high_and_no_material_difference() -> None:
    band = OperatingBand(lower=0.05, upper=0.15)
    counts = {
        _client("client-low"): BinomialCounts(x=0, n=200),
        _client("client-high"): BinomialCounts(x=190, n=200),
        _client("client-mid"): BinomialCounts(x=20, n=200),
    }
    decisions = bonferroni_fleet_sensitivity(counts, band, familywise_alpha=0.05)
    by_client = {decision.client_id: decision for decision in decisions}
    assert by_client[_client("client-low")].outcome is MismatchOutcome.LOW
    assert by_client[_client("client-high")].outcome is MismatchOutcome.HIGH
    assert by_client[_client("client-mid")].outcome is MismatchOutcome.NO_MATERIAL_DIFFERENCE
    for decision in decisions:
        assert decision.high_p_value is not None
        assert decision.low_p_value is not None


def test_bonferroni_fleet_sensitivity_skips_low_side_when_band_starts_at_zero() -> None:
    band = OperatingBand(lower=0.0, upper=0.15)
    counts = {_client("client-a"): BinomialCounts(x=0, n=200)}
    decisions = bonferroni_fleet_sensitivity(counts, band, familywise_alpha=0.05)
    assert decisions[0].outcome is not MismatchOutcome.LOW


def test_holm_directional_fleet_sensitivity_is_empty_for_no_clients() -> None:
    assert holm_directional_fleet_sensitivity({}, _BAND, familywise_alpha=0.05) == ()


def test_holm_directional_fleet_sensitivity_classifies_low_high_and_no_material_difference() -> (
    None
):
    band = OperatingBand(lower=0.05, upper=0.15)
    counts = {
        _client("client-low"): BinomialCounts(x=0, n=200),
        _client("client-high"): BinomialCounts(x=190, n=200),
        _client("client-mid"): BinomialCounts(x=20, n=200),
    }
    decisions = holm_directional_fleet_sensitivity(counts, band, familywise_alpha=0.05)
    by_client = {decision.client_id: decision for decision in decisions}
    assert by_client[_client("client-low")].outcome is MismatchOutcome.LOW
    assert by_client[_client("client-high")].outcome is MismatchOutcome.HIGH
    assert by_client[_client("client-mid")].outcome is MismatchOutcome.NO_MATERIAL_DIFFERENCE
    assert by_client[_client("client-low")].low_p_value is not None


def test_holm_directional_fleet_sensitivity_omits_low_hypothesis_for_zero_lower_band() -> None:
    band = OperatingBand(lower=0.0, upper=0.15)
    counts = {_client("client-a"): BinomialCounts(x=0, n=200)}
    decisions = holm_directional_fleet_sensitivity(counts, band, familywise_alpha=0.05)
    assert decisions[0].low_p_value is None
    assert decisions[0].outcome is not MismatchOutcome.LOW
