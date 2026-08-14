from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter

from fedcrg.thresholding.readiness import (
    CalibrationReadinessEvaluator,
    DeploymentDecision,
    ReadinessPlan,
    ReadinessPlanBuilder,
    ReadinessPlanCache,
    ReferenceMismatchEvaluator,
    ReferenceThreshold,
    bonferroni_fleet_sensitivity,
    build_reference_threshold,
    clopper_pearson_interval,
    continuity_diagnostics,
    coverage_probability,
    familywise_readiness_assurance,
    holm_directional_fleet_sensitivity,
    minimum_bidirectional_sample_count,
    reference_rank,
)
from fedcrg.types import (
    BinomialCounts,
    CalibrationReadinessState,
    ClientId,
    MismatchOutcome,
    OperatingBand,
)
from tests._fixtures import primary_protocol

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)
_BAND = OperatingBand(lower=0.005, upper=0.015)


def _client(name: str) -> ClientId:
    return _CLIENT_ID_ADAPTER.validate_python(name)


def test_reference_rank_rejects_nonpositive_sample_count() -> None:
    with pytest.raises(ValueError):
        reference_rank(0, 0.01)
    with pytest.raises(ValueError):
        reference_rank(-5, 0.01)


def test_build_reference_threshold_rejects_empty_client_arrays() -> None:
    with pytest.raises(ValueError):
        build_reference_threshold({_client("client-a"): np.array([])}, 0.01)


def test_coverage_probability_matches_readiness_plan_builder() -> None:
    plan = ReadinessPlanBuilder().build(2000, _BAND, 0.95)
    probability = coverage_probability(plan.rank, 2000, _BAND)
    assert probability == pytest.approx(plan.coverage_probability, abs=1e-12)
    assert 0.0 <= probability <= 1.0


def test_coverage_probability_rejects_rank_outside_sample_range() -> None:
    with pytest.raises(ValueError):
        coverage_probability(0, 2000, _BAND)
    with pytest.raises(ValueError):
        coverage_probability(2001, 2000, _BAND)


def test_readiness_plan_cache_precompute_detects_internal_inconsistency() -> None:
    cache = ReadinessPlanCache()
    key = cache.key(2000, _BAND, 0.95)
    consistent = cache.builder.build(2000, _BAND, 0.95)
    corrupted = consistent.model_copy(update={"rank": consistent.rank + 1})
    cache._plans[key] = corrupted
    with pytest.raises(RuntimeError):
        cache.precompute(2000, _BAND, 0.95)


def test_readiness_plan_cache_load_plans_rejects_formula_mismatch() -> None:
    cache = ReadinessPlanCache()
    reference = cache.builder.build(2000, _BAND, 0.95)
    tampered = ReadinessPlan(
        sample_count=2000,
        rank=reference.rank + 5,
        coverage_probability=reference.coverage_probability,
        state=reference.state,
        band=_BAND,
        assurance=0.95,
    )
    with pytest.raises(ValueError):
        cache.load_plans((tampered,))


def test_readiness_plan_cache_save_without_path_raises() -> None:
    cache = ReadinessPlanCache()
    with pytest.raises(ValueError):
        cache.save()


def test_readiness_plan_cache_load_rejects_non_list_payload(tmp_path: Path) -> None:
    path = tmp_path / "not_a_list.json"
    path.write_text(json.dumps({"unexpected": "object"}), encoding="utf-8")
    cache = ReadinessPlanCache()
    with pytest.raises(ValueError):
        cache.load(path)


def test_readiness_plan_cache_save_uses_explicit_path_override(tmp_path: Path) -> None:
    default_path = tmp_path / "default.json"
    override_path = tmp_path / "override.json"
    cache = ReadinessPlanCache(default_path)
    cache.precompute(2000, _BAND, 0.95)
    cache.save(override_path)
    assert override_path.exists()
    assert not default_path.exists()


def test_calibration_readiness_evaluator_rejects_nonfinite_scores() -> None:
    plan = ReadinessPlanBuilder().build(2000, _BAND, 0.95)
    scores = np.concatenate((np.full(1999, 0.5), [float("nan")]))
    with pytest.raises(ValueError):
        CalibrationReadinessEvaluator().evaluate(scores, plan)


def test_continuity_diagnostics_rejects_empty_scores() -> None:
    with pytest.raises(ValueError):
        continuity_diagnostics(np.array([]), 1)


def test_continuity_diagnostics_rejects_out_of_range_rank() -> None:
    with pytest.raises(ValueError):
        continuity_diagnostics(np.array([1.0, 2.0]), 5)
    with pytest.raises(ValueError):
        continuity_diagnostics(np.array([1.0, 2.0]), 0)


def test_continuity_diagnostics_reports_no_duplicates_for_distinct_scores() -> None:
    diagnostics = continuity_diagnostics(np.array([1.0, 2.0, 3.0]), 2)
    assert diagnostics.unique_score_fraction == 1.0
    assert diagnostics.duplicate_count == 0
    assert diagnostics.selected_threshold_multiplicity == 1
    assert diagnostics.minimum_positive_spacing == pytest.approx(1.0)


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
    with pytest.raises(ValueError):
        clopper_pearson_interval(BinomialCounts(x=1, n=10), 0.0)
    with pytest.raises(ValueError):
        clopper_pearson_interval(BinomialCounts(x=1, n=10), 1.0)


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


def test_reference_mismatch_evaluator_rejects_nonfinite_scores() -> None:
    evaluator = ReferenceMismatchEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(
            scores=np.array([1.0, float("nan")]),
            reference_threshold=0.5,
            band=_BAND,
            confidence=0.95,
        )


def test_reference_mismatch_evaluator_rejects_nonfinite_threshold() -> None:
    evaluator = ReferenceMismatchEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(
            scores=np.array([1.0]),
            reference_threshold=float("inf"),
            band=_BAND,
            confidence=0.95,
        )


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


def test_deployment_decision_flags_assumption_violation_on_calibration_tie() -> None:
    plan = ReadinessPlanBuilder().build(2000, _BAND, 0.95)
    values = np.linspace(0.5, 1.0, 2000)
    values[plan.rank - 8 : plan.rank + 7] = values[plan.rank - 1]
    readiness = CalibrationReadinessEvaluator().evaluate(values, plan)
    assert readiness.plan.state is CalibrationReadinessState.READY
    assert readiness.tie_count > 1

    reference = ReferenceThreshold(
        value=0.75, rank=2, sample_count=4, client_count=2, samples_per_client=2
    )
    mismatch = ReferenceMismatchEvaluator().evaluate(
        scores=np.linspace(0.8, 0.9, 736),
        reference_threshold=reference.value,
        band=_BAND,
        confidence=primary_protocol().mismatch_confidence,
    )
    assert mismatch.outcome is MismatchOutcome.HIGH

    decision = DeploymentDecision().decide(
        reference=reference,
        readiness=readiness,
        mismatch=mismatch,
        reject_calibration_ties=True,
    )
    assert decision.state.value == "CALIBRATION_ASSUMPTION_VIOLATION"
    assert decision.threshold == reference.value
    assert decision.tie_count == readiness.tie_count

    admitted = DeploymentDecision().decide(
        reference=reference,
        readiness=readiness,
        mismatch=mismatch,
        reject_calibration_ties=False,
    )
    assert admitted.state.value == "LOCAL_PERSONALIZE"
    assert admitted.threshold == readiness.threshold


def test_reference_rank_locked_examples_use_ceiling_formula() -> None:
    for sample_count, alpha in ((100, 0.01), (736, 0.01), (1, 0.5)):
        expected = min(sample_count, math.ceil((sample_count + 1) * (1.0 - alpha)))
        assert reference_rank(sample_count, alpha) == expected
