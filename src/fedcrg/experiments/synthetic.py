"""Statistical kernels for the pre-registered synthetic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.stats import binom, gamma, lognorm, norm

from fedcrg.core.enums import (
    ContaminationDirection,
    ExperimentCode,
    SyntheticDistribution,
)
from fedcrg.core.types import OperatingBand
from fedcrg.protocol.mismatch import clopper_pearson_interval
from fedcrg.protocol.readiness import ReadinessPlanBuilder


@dataclass(frozen=True, slots=True)
class SyntheticCoverageResult:
    experiment: ExperimentCode
    condition: SyntheticDistribution | ContaminationDirection
    sample_count: int
    exact_probability: float
    empirical_probability: float
    repetitions: int
    accepted: bool | None


@dataclass(frozen=True, slots=True)
class MismatchPowerResult:
    sample_count: int
    true_fpr: float
    declaration_probability: float


def draw_distribution(
    rng: np.random.Generator,
    distribution: SyntheticDistribution,
    size: int,
) -> np.ndarray:
    if size <= 0:
        raise ValueError("Synthetic sample size must be positive")
    if distribution is SyntheticDistribution.NORMAL:
        return rng.normal(size=size)
    if distribution is SyntheticDistribution.LOGNORMAL:
        return rng.lognormal(mean=0.0, sigma=1.0, size=size)
    if distribution is SyntheticDistribution.GAMMA_SHAPE_2:
        return rng.gamma(shape=2.0, scale=1.0, size=size)
    if distribution is SyntheticDistribution.NORMAL_MIXTURE:
        component = rng.random(size) < 0.1
        values = rng.normal(size=size)
        values[component] = rng.normal(
            loc=3.0,
            scale=1.0,
            size=int(component.sum()),
        )
        return values
    raise AssertionError(f"Unhandled synthetic distribution: {distribution}")


def distribution_cdf(
    distribution: SyntheticDistribution,
    threshold: float,
) -> float:
    if distribution is SyntheticDistribution.NORMAL:
        return float(norm.cdf(threshold))
    if distribution is SyntheticDistribution.LOGNORMAL:
        return float(lognorm.cdf(threshold, s=1.0, scale=1.0))
    if distribution is SyntheticDistribution.GAMMA_SHAPE_2:
        return float(gamma.cdf(threshold, a=2.0, scale=1.0))
    if distribution is SyntheticDistribution.NORMAL_MIXTURE:
        return float(0.9 * norm.cdf(threshold) + 0.1 * norm.cdf(threshold, loc=3.0, scale=1.0))
    raise AssertionError(f"Unhandled synthetic distribution: {distribution}")


def iid_readiness_validation(
    experiment: ExperimentCode,
    distribution: SyntheticDistribution,
    sample_count: int,
    repetitions: int,
    *,
    alpha: float,
    rho: float,
    assurance: float,
    seed: int,
) -> SyntheticCoverageResult:
    band = OperatingBand(
        max(0.0, alpha * (1.0 - rho)),
        min(1.0, alpha * (1.0 + rho)),
    )
    plan = ReadinessPlanBuilder().build(sample_count, band, assurance)
    rng = np.random.Generator(np.random.PCG64(seed))
    inside = 0
    for _ in range(repetitions):
        values = np.sort(
            draw_distribution(rng, distribution, sample_count),
            kind="stable",
        )
        threshold = float(values[plan.rank - 1])
        future_fpr = 1.0 - distribution_cdf(distribution, threshold)
        inside += int(band.lower <= future_fpr <= band.upper)
    empirical = inside / repetitions
    tolerance = max(
        0.005,
        4.0 * sqrt(plan.coverage_probability * (1.0 - plan.coverage_probability) / repetitions),
    )
    return SyntheticCoverageResult(
        experiment=experiment,
        condition=distribution,
        sample_count=sample_count,
        exact_probability=plan.coverage_probability,
        empirical_probability=empirical,
        repetitions=repetitions,
        accepted=abs(empirical - plan.coverage_probability) <= tolerance,
    )


def contamination_validation(
    fraction: float,
    direction: ContaminationDirection,
    repetitions: int,
    *,
    sample_count: int,
    seed: int,
) -> SyntheticCoverageResult:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Contamination fraction must be in [0,1]")
    band = OperatingBand(0.005, 0.015)
    plan = ReadinessPlanBuilder().build(sample_count, band, 0.95)
    rng = np.random.Generator(np.random.PCG64(seed))
    inside = 0
    contamination_count = int(round(fraction * sample_count))
    location = 3.0 if direction is ContaminationDirection.HIGH else -3.0
    for _ in range(repetitions):
        values = rng.normal(size=sample_count)
        if contamination_count:
            indices = rng.choice(
                sample_count,
                size=contamination_count,
                replace=False,
            )
            values[indices] = rng.normal(
                loc=location,
                scale=1.0,
                size=contamination_count,
            )
        threshold = float(np.sort(values, kind="stable")[plan.rank - 1])
        future_fpr = 1.0 - float(norm.cdf(threshold))
        inside += int(band.lower <= future_fpr <= band.upper)
    return SyntheticCoverageResult(
        experiment=ExperimentCode.S5,
        condition=direction,
        sample_count=sample_count,
        exact_probability=plan.coverage_probability,
        empirical_probability=inside / repetitions,
        repetitions=repetitions,
        accepted=None,
    )


def exact_mismatch_power(
    sample_count: int,
    true_fpr: float,
) -> MismatchPowerResult:
    if sample_count <= 0 or not 0.0 <= true_fpr <= 1.0:
        raise ValueError("Mismatch power requires n>0 and true_fpr in [0,1]")
    band = OperatingBand(0.005, 0.015)
    low_counts: list[int] = []
    high_counts: list[int] = []
    for exceedances in range(sample_count + 1):
        interval = clopper_pearson_interval(exceedances, sample_count, 0.95)
        if interval.upper < band.lower:
            low_counts.append(exceedances)
        elif interval.lower > band.upper:
            high_counts.append(exceedances)
    probability = 0.0
    if low_counts:
        probability += float(binom.cdf(max(low_counts), sample_count, true_fpr))
    if high_counts:
        probability += float(binom.sf(min(high_counts) - 1, sample_count, true_fpr))
    return MismatchPowerResult(sample_count, true_fpr, probability)
