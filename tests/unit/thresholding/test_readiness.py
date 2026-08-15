from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter

from fedcrg.thresholding.readiness import (
    CalibrationReadiness,
    CalibrationReadinessEvaluator,
    ReadinessPlan,
    ReadinessPlanBuilder,
    ReadinessPlanCache,
    ReferenceMismatchEvaluator,
    build_reference_threshold,
    continuity_diagnostics,
    coverage_probability,
    minimum_bidirectional_sample_count,
    reference_rank,
)
from fedcrg.types import (
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


def test_reference_rank_locked_examples_use_ceiling_formula() -> None:
    for sample_count, alpha in ((100, 0.01), (736, 0.01), (1, 0.5)):
        expected = min(sample_count, math.ceil((sample_count + 1) * (1.0 - alpha)))
        assert reference_rank(sample_count, alpha) == expected


def test_reference_threshold_pools_equal_counts_across_clients() -> None:
    scores = {
        _client("client-a"): np.zeros(2),
        _client("client-b"): np.ones(2),
    }
    reference = build_reference_threshold(scores, 0.01)
    assert reference.sample_count == 4
    assert reference.client_count == 2
    assert reference.samples_per_client == 2
    assert reference.rank == 4
    assert reference.value == 1.0


def test_reference_threshold_rejects_unequal_contributions() -> None:
    scores = {
        _client("client-a"): np.zeros(2),
        _client("client-b"): np.ones(3),
    }
    with pytest.raises(ValueError):
        build_reference_threshold(scores, 0.01)


def test_reference_threshold_rejects_empty_pool() -> None:
    with pytest.raises(ValueError):
        build_reference_threshold({}, 0.01)


def test_reference_threshold_rejects_empty_client_arrays() -> None:
    scores = {_client("client-a"): np.array([])}
    with pytest.raises(ValueError):
        build_reference_threshold(scores, 0.01)


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


def test_readiness_plan_roundtrip_via_model() -> None:
    protocol = primary_protocol()
    plan = ReadinessPlanBuilder().build(2000, protocol.band, protocol.readiness_assurance)
    decoded = ReadinessPlan.model_validate_json(plan.model_dump_json())
    assert decoded == plan
    assert decoded.state is CalibrationReadinessState.READY
    assert isinstance(decoded.band, OperatingBand)


def test_readiness_evaluator_can_fail_a_plan() -> None:
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


def test_readiness_evaluator_rejects_size_mismatch() -> None:
    protocol = primary_protocol()
    plan = ReadinessPlanBuilder().build(2000, protocol.band, protocol.readiness_assurance)
    evaluator = CalibrationReadinessEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(np.linspace(0.0, 1.0, 100), plan)
    with pytest.raises(ValueError):
        evaluator.evaluate(np.array([]), plan)


def test_readiness_evaluator_rejects_nonfinite_scores() -> None:
    plan = ReadinessPlanBuilder().build(2000, _BAND, 0.95)
    scores = np.concatenate((np.full(1999, 0.5), [float("nan")]))
    evaluator = CalibrationReadinessEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(scores, plan)


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


def test_runtime_lookup_never_materializes_missing_readiness_plan(tmp_path: Path) -> None:
    path = tmp_path / "readiness_plans.json"
    cache = ReadinessPlanCache(path)
    band = OperatingBand(lower=0.005, upper=0.015)

    with pytest.raises(FileNotFoundError):
        cache.require(2000, band, 0.95)
    assert not path.exists()


def test_precompute_persists_then_runtime_requires_exact_contract(tmp_path: Path) -> None:
    path = tmp_path / "readiness_plans.json"
    band = OperatingBand(lower=0.005, upper=0.015)
    writer = ReadinessPlanCache(path)
    plan = writer.precompute(2000, band, 0.95)
    writer.save()

    reader = ReadinessPlanCache(path)
    assert reader.require(2000, band, 0.95) == plan
    with pytest.raises(FileNotFoundError):
        reader.require(2000, band, 0.99)


def test_minimum_bidirectional_sample_count_is_locked() -> None:
    protocol = primary_protocol()
    minimum = minimum_bidirectional_sample_count(protocol.band.lower, protocol.mismatch_confidence)
    assert minimum == 736


def test_mismatch_evaluator_detects_high_mismatch() -> None:
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


def test_mismatch_evaluator_returns_insufficient_evidence_below_minimum() -> None:
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


def test_mismatch_evaluator_rejects_empty_scores() -> None:
    protocol = primary_protocol()
    evaluator = ReferenceMismatchEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(
            scores=np.array([]),
            reference_threshold=0.75,
            band=protocol.band,
            confidence=protocol.mismatch_confidence,
        )


def test_mismatch_evaluator_rejects_nonfinite_scores() -> None:
    evaluator = ReferenceMismatchEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(
            scores=np.array([1.0, float("nan")]),
            reference_threshold=0.5,
            band=_BAND,
            confidence=0.95,
        )


def test_mismatch_evaluator_rejects_nonfinite_threshold() -> None:
    evaluator = ReferenceMismatchEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(
            scores=np.array([1.0]),
            reference_threshold=float("inf"),
            band=_BAND,
            confidence=0.95,
        )
