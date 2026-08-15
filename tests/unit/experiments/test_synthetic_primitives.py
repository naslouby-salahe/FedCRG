"""Tests for the distribution sampling, CDF, and coverage-stress primitives used by the synthetic experiments."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import gamma, lognorm, norm

from fedcrg.config import Study
from fedcrg.experiments.analyses import (
    calibration_shift_stress,
    contamination_validation,
    distribution_cdf,
    draw_distribution,
    exact_mismatch_power,
    iid_readiness_validation,
    temporal_dependence_stress,
)
from fedcrg.types import (
    ContaminationDirection,
    ExperimentId,
    OperatingBand,
    SyntheticDistribution,
)


@pytest.fixture(scope="module")
def synthetic_config():
    return Study.load().resolve(ExperimentId.READINESS_THEOREM)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(123))


def test_draw_distribution_normal_has_the_requested_size(rng, synthetic_config) -> None:
    values = draw_distribution(rng, SyntheticDistribution.NORMAL, 500, synthetic_config.synthetic)
    assert values.shape == (500,)


def test_draw_distribution_lognormal_is_non_negative(rng, synthetic_config) -> None:
    values = draw_distribution(
        rng, SyntheticDistribution.LOGNORMAL, 500, synthetic_config.synthetic
    )
    assert values.shape == (500,)
    assert np.all(values >= 0.0)


def test_draw_distribution_gamma_is_non_negative(rng, synthetic_config) -> None:
    values = draw_distribution(
        rng, SyntheticDistribution.GAMMA_SHAPE_2, 500, synthetic_config.synthetic
    )
    assert values.shape == (500,)
    assert np.all(values >= 0.0)


def test_draw_distribution_normal_mixture_has_the_requested_size(rng, synthetic_config) -> None:
    values = draw_distribution(
        rng, SyntheticDistribution.NORMAL_MIXTURE, 2000, synthetic_config.synthetic
    )
    assert values.shape == (2000,)
    assert np.isfinite(values).all()


def test_draw_distribution_rejects_non_positive_size(rng, synthetic_config) -> None:
    with pytest.raises(ValueError):
        draw_distribution(rng, SyntheticDistribution.NORMAL, 0, synthetic_config.synthetic)


def test_draw_distribution_rejects_an_unhandled_distribution(rng, synthetic_config) -> None:
    with pytest.raises(AssertionError):
        draw_distribution(rng, "not_a_distribution", 10, synthetic_config.synthetic)


def test_draw_distribution_normal_mixture_with_zero_weight_has_no_mixture_component(
    rng, synthetic_config
) -> None:
    zero_weight_synthetic = synthetic_config.synthetic.model_copy(update={"mixture_weight": 0.0})
    values = draw_distribution(
        rng, SyntheticDistribution.NORMAL_MIXTURE, 200, zero_weight_synthetic
    )
    assert values.shape == (200,)
    assert np.isfinite(values).all()


def test_distribution_cdf_rejects_an_unhandled_distribution(synthetic_config) -> None:
    with pytest.raises(AssertionError):
        distribution_cdf("not_a_distribution", 0.5, synthetic_config.synthetic)


def test_distribution_cdf_normal_matches_scipy(synthetic_config) -> None:
    value = distribution_cdf(SyntheticDistribution.NORMAL, 0.5, synthetic_config.synthetic)
    assert value == pytest.approx(float(norm.cdf(0.5)))


def test_distribution_cdf_lognormal_matches_scipy(synthetic_config) -> None:
    synthetic = synthetic_config.synthetic
    value = distribution_cdf(SyntheticDistribution.LOGNORMAL, 1.0, synthetic)
    expected = float(
        lognorm.cdf(1.0, s=synthetic.lognormal_sigma, scale=float(np.exp(synthetic.lognormal_mean)))
    )
    assert value == pytest.approx(expected)


def test_distribution_cdf_gamma_matches_scipy(synthetic_config) -> None:
    synthetic = synthetic_config.synthetic
    value = distribution_cdf(SyntheticDistribution.GAMMA_SHAPE_2, 1.0, synthetic)
    expected = float(gamma.cdf(1.0, a=synthetic.gamma_shape, scale=synthetic.gamma_scale))
    assert value == pytest.approx(expected)


def test_distribution_cdf_normal_mixture_matches_manual_formula(synthetic_config) -> None:
    synthetic = synthetic_config.synthetic
    value = distribution_cdf(SyntheticDistribution.NORMAL_MIXTURE, 0.5, synthetic)
    expected = (1.0 - synthetic.mixture_weight) * float(
        norm.cdf(0.5)
    ) + synthetic.mixture_weight * float(
        norm.cdf(0.5, loc=synthetic.mixture_shift, scale=synthetic.mixture_scale)
    )
    assert value == pytest.approx(expected)


def test_iid_readiness_validation_produces_a_coverage_result_near_the_exact_probability(
    synthetic_config,
) -> None:
    result = iid_readiness_validation(
        ExperimentId.READINESS_THEOREM,
        SyntheticDistribution.NORMAL,
        200,
        400,
        alpha=synthetic_config.protocol.alpha,
        rho=synthetic_config.protocol.rho,
        assurance=synthetic_config.protocol.readiness_assurance,
        seed=17,
        synthetic=synthetic_config.synthetic,
    )
    assert result.sample_count == 200
    assert result.repetitions == 400
    assert result.condition is SyntheticDistribution.NORMAL
    assert 0.0 <= result.empirical_probability <= 1.0
    assert 0.0 <= result.exact_probability <= 1.0
    assert result.accepted is not None


def test_contamination_validation_with_zero_fraction_behaves_like_pure_iid(
    synthetic_config,
) -> None:
    band = synthetic_config.protocol.band
    result = contamination_validation(
        0.0,
        ContaminationDirection.HIGH,
        300,
        sample_count=200,
        seed=5,
        band=band,
        assurance=synthetic_config.protocol.readiness_assurance,
        synthetic=synthetic_config.synthetic,
    )
    assert result.condition is ContaminationDirection.HIGH
    assert result.sample_count == 200
    assert result.repetitions == 300
    assert result.accepted is None
    assert 0.0 <= result.empirical_probability <= 1.0


def test_contamination_validation_low_direction_shifts_scores_downward(synthetic_config) -> None:
    band = synthetic_config.protocol.band
    high = contamination_validation(
        0.1,
        ContaminationDirection.HIGH,
        200,
        sample_count=300,
        seed=99,
        band=band,
        assurance=synthetic_config.protocol.readiness_assurance,
        synthetic=synthetic_config.synthetic,
    )
    low = contamination_validation(
        0.1,
        ContaminationDirection.LOW,
        200,
        sample_count=300,
        seed=99,
        band=band,
        assurance=synthetic_config.protocol.readiness_assurance,
        synthetic=synthetic_config.synthetic,
    )
    assert high.condition is ContaminationDirection.HIGH
    assert low.condition is ContaminationDirection.LOW


def test_contamination_validation_rejects_out_of_range_fraction(synthetic_config) -> None:
    with pytest.raises(ValueError):
        contamination_validation(
            1.5,
            ContaminationDirection.HIGH,
            10,
            sample_count=100,
            seed=1,
            band=synthetic_config.protocol.band,
            assurance=synthetic_config.protocol.readiness_assurance,
            synthetic=synthetic_config.synthetic,
        )


def test_exact_mismatch_power_declaration_probability_is_high_when_true_fpr_is_far_from_band() -> (
    None
):
    band = OperatingBand(lower=0.005, upper=0.015)
    result = exact_mismatch_power(50, 0.5, band, 0.95)
    assert result.sample_count == 50
    assert result.true_fpr == 0.5
    assert result.declaration_probability > 0.99


def test_exact_mismatch_power_declaration_probability_is_low_when_true_fpr_matches_band() -> None:
    band = OperatingBand(lower=0.005, upper=0.015)
    result = exact_mismatch_power(50, 0.01, band, 0.95)
    assert 0.0 <= result.declaration_probability < 0.5


def test_exact_mismatch_power_accumulates_both_low_and_high_declaration_regions() -> None:
    band = OperatingBand(lower=0.005, upper=0.015)
    result = exact_mismatch_power(800, 0.01, band, 0.95)
    assert 0.0 <= result.declaration_probability <= 1.0


def test_exact_mismatch_power_has_no_high_declaration_region_when_band_upper_is_one() -> None:
    band = OperatingBand(lower=0.0001, upper=1.0)
    result = exact_mismatch_power(50, 0.01, band, 0.95)
    assert 0.0 <= result.declaration_probability <= 1.0


def test_exact_mismatch_power_rejects_invalid_sample_count() -> None:
    band = OperatingBand(lower=0.005, upper=0.015)
    with pytest.raises(ValueError):
        exact_mismatch_power(0, 0.01, band, 0.95)


def test_exact_mismatch_power_rejects_invalid_true_fpr() -> None:
    band = OperatingBand(lower=0.005, upper=0.015)
    with pytest.raises(ValueError):
        exact_mismatch_power(50, 1.5, band, 0.95)


def test_temporal_dependence_stress_returns_a_robustness_cell(synthetic_config) -> None:
    cell = temporal_dependence_stress(
        0.3,
        200,
        150,
        synthetic_config.protocol.band,
        synthetic_config.protocol.readiness_assurance,
        seed=31,
    )
    assert cell.value == 0.3
    assert cell.repetitions == 150
    assert 0.0 <= cell.coverage <= 1.0


def test_temporal_dependence_stress_rejects_phi_outside_open_unit_interval() -> None:
    band = OperatingBand(lower=0.005, upper=0.015)
    with pytest.raises(ValueError):
        temporal_dependence_stress(1.0, 100, 10, band, 0.95, seed=1)


def test_calibration_shift_stress_returns_a_robustness_cell(synthetic_config) -> None:
    cell = calibration_shift_stress(
        0.5,
        150,
        synthetic_config.protocol.band,
        synthetic_config.protocol.readiness_assurance,
        seed=13,
        sample_count=200,
    )
    assert cell.value == 0.5
    assert cell.repetitions == 150
    assert 0.0 <= cell.coverage <= 1.0


def test_calibration_shift_stress_large_shift_reduces_coverage_relative_to_no_shift(
    synthetic_config,
) -> None:
    band = synthetic_config.protocol.band
    assurance = synthetic_config.protocol.readiness_assurance
    baseline = calibration_shift_stress(0.0, 500, band, assurance, seed=71, sample_count=300)
    shifted = calibration_shift_stress(5.0, 500, band, assurance, seed=71, sample_count=300)
    assert shifted.coverage <= baseline.coverage
