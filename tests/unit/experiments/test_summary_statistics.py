"""Tests for descriptive, bootstrap, and split-sensitivity summary statistics."""

from __future__ import annotations

import numpy as np
import pytest

from fedcrg.experiments.analyses import (
    FederationResultRecord,
    contaminate_benign_scores,
    describe,
    paired_model_seed_bootstrap,
    source_order_blocks,
    split_sensitivity,
    split_sensitivity_summary,
    summarize_state_stability,
    summarize_threshold_stability,
)
from fedcrg.types import DatasetId, DecisionState, ExperimentId, PolicyId


def test_describe_computes_expected_summary() -> None:
    summary = describe((1.0, 2.0, 3.0, 4.0))
    assert summary.mean == 2.5
    assert summary.median == 2.5
    assert summary.minimum == 1.0
    assert summary.maximum == 4.0
    assert summary.standard_deviation == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))


def test_describe_single_value_has_zero_standard_deviation() -> None:
    summary = describe((5.0,))
    assert summary.standard_deviation == 0.0


def test_describe_rejects_empty_values() -> None:
    with pytest.raises(ValueError):
        describe(())


def test_describe_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        describe((1.0, float("nan")))


def test_split_sensitivity_summary_matches_numpy_percentiles() -> None:
    values = tuple(float(v) for v in range(1, 101))
    summary = split_sensitivity_summary(values, (5, 25, 50, 75, 95))
    p5, p25, p50, p75, p95 = np.percentile(values, (5, 25, 50, 75, 95))
    assert summary.median == pytest.approx(p50)
    assert summary.iqr == pytest.approx(p75 - p25)
    assert summary.p05 == pytest.approx(p5)
    assert summary.p95 == pytest.approx(p95)


def test_split_sensitivity_summary_rejects_empty_values() -> None:
    with pytest.raises(ValueError):
        split_sensitivity_summary((), (5, 25, 50, 75, 95))


def test_paired_model_seed_bootstrap_zero_difference_for_identical_vectors() -> None:
    values = (1.0, 2.0, 3.0, 4.0, 5.0)
    result = paired_model_seed_bootstrap(
        values, values, replicates=200, seed=7, ci_percentiles=(2.5, 97.5)
    )
    assert result.observed_difference == 0.0
    assert result.lower == 0.0
    assert result.upper == 0.0
    assert result.replicates == 200
    assert result.seed == 7


def test_paired_model_seed_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    method = (1.0, 2.0, 3.0, 4.0, 5.0)
    comparator = (2.0, 2.0, 2.0, 2.0, 2.0)
    first = paired_model_seed_bootstrap(
        method, comparator, replicates=500, seed=11, ci_percentiles=(5, 95)
    )
    second = paired_model_seed_bootstrap(
        method, comparator, replicates=500, seed=11, ci_percentiles=(5, 95)
    )
    assert first == second
    assert first.observed_difference == pytest.approx(1.0)
    assert first.lower <= first.observed_difference <= first.upper


def test_paired_model_seed_bootstrap_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        paired_model_seed_bootstrap(
            (1.0, 2.0), (1.0,), replicates=10, seed=1, ci_percentiles=(5, 95)
        )


def test_paired_model_seed_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        paired_model_seed_bootstrap((), (), replicates=10, seed=1, ci_percentiles=(5, 95))


def test_summarize_threshold_stability_rejects_empty_values() -> None:
    with pytest.raises(ValueError):
        summarize_threshold_stability((), (25, 75))


def test_summarize_threshold_stability_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        summarize_threshold_stability((1.0, float("inf")), (25, 75))


def test_summarize_state_stability_rejects_empty_states() -> None:
    with pytest.raises(ValueError):
        summarize_state_stability(())


def test_summarize_state_stability_all_same_state_has_no_transitions() -> None:
    states = (DecisionState.REFERENCE_RETAINED,) * 5
    stability = summarize_state_stability(states)
    assert stability.transition_count == 0
    assert stability.transition_frequency == 0.0
    assert stability.count == 5
    frequencies = {row.state: row.frequency for row in stability.state_frequencies}
    assert frequencies[DecisionState.REFERENCE_RETAINED] == 1.0
    assert set(frequencies) == set(DecisionState)


def test_stability_summaries_report_explicit_values() -> None:
    threshold = summarize_threshold_stability((1.0, 2.0, 3.0), (25, 75))
    assert threshold.count == 3
    assert threshold.iqr == 1.0
    states = summarize_state_stability(
        (
            DecisionState.REFERENCE_RETAINED,
            DecisionState.REFERENCE_RETAINED,
            DecisionState.PERSONALIZED,
            DecisionState.PERSONALIZED,
        )
    )
    assert states.transition_count == 1
    assert states.transition_frequency == 1 / 3


def _record(
    *,
    model_seed: int,
    policy: PolicyId,
    calibration_seed: int = 1,
    mebe: float = 0.1,
    high_excess: float = 0.2,
    band_violation_rate: float = 0.0,
    attack_balanced_macro_tpr: float | None = 0.5,
) -> FederationResultRecord:
    return FederationResultRecord(
        run_id="run",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        dataset_id=DatasetId.NBAIOT,
        model_seed=model_seed,
        calibration_seed=calibration_seed,
        policy=policy,
        mebe=mebe,
        high_excess=high_excess,
        band_violation_rate=band_violation_rate,
        attack_balanced_macro_tpr=attack_balanced_macro_tpr,
    )


def test_split_sensitivity_groups_by_seed_policy_metric_and_sorts() -> None:
    records = (
        _record(model_seed=2, policy=PolicyId.FEDCRG, mebe=0.1),
        _record(model_seed=2, policy=PolicyId.FEDCRG, mebe=0.3),
        _record(model_seed=1, policy=PolicyId.FEDCRG, mebe=0.2),
        _record(model_seed=1, policy=PolicyId.FEDCRG, mebe=0.4),
    )
    percentiles = (5, 25, 50, 75, 95)
    rows = split_sensitivity(records, percentiles)
    mebe_rows = [row for row in rows if row.metric == "mebe"]
    assert [row.model_seed for row in mebe_rows] == [1, 2]
    assert all(row.calibration_split_count == 2 for row in rows)
    assert [row.model_seed for row in rows] == sorted(row.model_seed for row in rows)


def test_split_sensitivity_skips_none_valued_metrics() -> None:
    records = (
        _record(model_seed=1, policy=PolicyId.FEDCRG, attack_balanced_macro_tpr=None),
        _record(model_seed=1, policy=PolicyId.FEDCRG, attack_balanced_macro_tpr=None),
    )
    percentiles = (5, 25, 50, 75, 95)
    rows = split_sensitivity(records, percentiles)
    assert "attack_balanced_macro_tpr" not in {row.metric for row in rows}
    assert {row.metric for row in rows} == {"mebe", "high_excess"}


def test_split_sensitivity_of_empty_records_is_empty() -> None:
    assert split_sensitivity((), (5, 25, 50, 75, 95)) == ()


def test_source_order_blocks_splits_into_requested_block_count() -> None:
    scores = np.arange(10, dtype=np.float64)
    blocks = source_order_blocks(scores, 3)
    assert len(blocks) == 3
    assert sum(len(block) for block in blocks) == 10
    assert np.array_equal(np.concatenate(blocks), scores)


def test_source_order_blocks_rejects_non_positive_block_count() -> None:
    with pytest.raises(ValueError):
        source_order_blocks(np.arange(5, dtype=np.float64), 0)


def test_source_order_blocks_rejects_fewer_rows_than_blocks() -> None:
    with pytest.raises(ValueError):
        source_order_blocks(np.arange(2, dtype=np.float64), 5)


def test_contaminate_benign_scores_zero_fraction_is_a_no_op() -> None:
    benign = np.array([1.0, 2.0, 3.0])
    attacks = np.array([100.0, 200.0])
    result = contaminate_benign_scores(benign, attacks, 0.0, seed=1)
    assert np.array_equal(result, benign)


def test_contaminate_benign_scores_replaces_expected_count() -> None:
    benign = np.zeros(10)
    attacks = np.full(10, 999.0)
    result = contaminate_benign_scores(benign, attacks, 0.3, seed=42)
    assert np.count_nonzero(result == 999.0) == 3
    assert np.count_nonzero(result == 0.0) == 7


def test_contaminate_benign_scores_rejects_negative_fraction() -> None:
    with pytest.raises(ValueError):
        contaminate_benign_scores(np.zeros(4), np.zeros(4), -0.1, seed=1)


def test_contaminate_benign_scores_rejects_fraction_above_one() -> None:
    with pytest.raises(ValueError):
        contaminate_benign_scores(np.zeros(4), np.zeros(4), 1.1, seed=1)


def test_contaminate_benign_scores_rejects_undersized_attack_cache() -> None:
    with pytest.raises(ValueError):
        contaminate_benign_scores(np.zeros(10), np.zeros(2), 0.5, seed=1)
