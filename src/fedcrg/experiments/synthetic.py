"""Pre-registered synthetic experiments S1-S6."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.stats import binom, gamma, lognorm, norm

from fedcrg.core.types import OperatingBand
from fedcrg.protocol.mismatch import clopper_pearson_interval
from fedcrg.protocol.readiness import ReadinessPlanBuilder


@dataclass(frozen=True, slots=True)
class SyntheticCoverageResult:
    experiment: str
    condition: str
    sample_count: int
    exact_probability: float
    empirical_probability: float
    repetitions: int
    accepted: bool


@dataclass(frozen=True, slots=True)
class MismatchPowerResult:
    sample_count: int
    true_fpr: float
    declaration_probability: float


def _draw(rng: np.random.Generator, distribution: str, size: int) -> np.ndarray:
    if distribution == "normal":
        return rng.normal(size=size)
    if distribution == "lognormal":
        return rng.lognormal(mean=0.0, sigma=1.0, size=size)
    if distribution == "gamma2":
        return rng.gamma(shape=2.0, scale=1.0, size=size)
    if distribution == "normal_mixture":
        component = rng.random(size) < 0.1
        values = rng.normal(size=size)
        values[component] = rng.normal(loc=3.0, scale=1.0, size=int(component.sum()))
        return values
    raise ValueError(f"Unknown synthetic distribution {distribution}")


def _cdf(distribution: str, threshold: float) -> float:
    if distribution == "normal": return float(norm.cdf(threshold))
    if distribution == "lognormal": return float(lognorm.cdf(threshold, s=1.0, scale=1.0))
    if distribution == "gamma2": return float(gamma.cdf(threshold, a=2.0, scale=1.0))
    if distribution == "normal_mixture": return float(0.9 * norm.cdf(threshold) + 0.1 * norm.cdf(threshold, loc=3.0, scale=1.0))
    raise ValueError(distribution)


def iid_readiness_validation(
    distribution: str,
    sample_count: int,
    repetitions: int = 10000,
    *,
    alpha: float = 0.01,
    rho: float = 0.5,
    assurance: float = 0.95,
    seed: int = 123456,
) -> SyntheticCoverageResult:
    band = OperatingBand(max(0.0, alpha * (1.0 - rho)), min(1.0, alpha * (1.0 + rho)))
    plan = ReadinessPlanBuilder().build(sample_count, band, assurance)
    rng = np.random.Generator(np.random.PCG64(seed))
    inside = 0
    for _ in range(repetitions):
        values = np.sort(_draw(rng, distribution, sample_count), kind="stable")
        threshold = float(values[plan.rank - 1])
        future_fpr = 1.0 - _cdf(distribution, threshold)
        inside += int(band.lower <= future_fpr <= band.upper)
    empirical = inside / repetitions
    tolerance = max(0.005, 4.0 * sqrt(plan.coverage_probability * (1.0 - plan.coverage_probability) / repetitions))
    return SyntheticCoverageResult(
        "S1",
        distribution,
        sample_count,
        plan.coverage_probability,
        empirical,
        repetitions,
        abs(empirical - plan.coverage_probability) <= tolerance,
    )


def contamination_validation(
    fraction: float,
    direction: str,
    repetitions: int = 10000,
    *,
    sample_count: int = 2000,
    seed: int = 123456,
) -> SyntheticCoverageResult:
    band = OperatingBand(0.005, 0.015)
    plan = ReadinessPlanBuilder().build(sample_count, band, 0.95)
    rng = np.random.Generator(np.random.PCG64(seed))
    inside = 0
    contamination_count = int(round(fraction * sample_count))
    location = 3.0 if direction == "high" else -3.0
    for _ in range(repetitions):
        values = rng.normal(size=sample_count)
        if contamination_count:
            indices = rng.choice(sample_count, size=contamination_count, replace=False)
            values[indices] = rng.normal(loc=location, scale=1.0, size=contamination_count)
        threshold = float(np.sort(values)[plan.rank - 1])
        future_fpr = 1.0 - norm.cdf(threshold)
        inside += int(band.lower <= future_fpr <= band.upper)
    return SyntheticCoverageResult("S5", direction, sample_count, plan.coverage_probability, inside / repetitions, repetitions, True)


def exact_mismatch_power(sample_count: int, true_fpr: float) -> MismatchPowerResult:
    band = OperatingBand(0.005, 0.015)
    low_counts: list[int] = []
    high_counts: list[int] = []
    for x in range(sample_count + 1):
        interval = clopper_pearson_interval(x, sample_count, 0.95)
        if interval.upper < band.lower:
            low_counts.append(x)
        elif interval.lower > band.upper:
            high_counts.append(x)
    probability = 0.0
    if low_counts:
        probability += float(binom.cdf(max(low_counts), sample_count, true_fpr))
    if high_counts:
        probability += float(binom.sf(min(high_counts) - 1, sample_count, true_fpr))
    return MismatchPowerResult(sample_count, true_fpr, probability)
