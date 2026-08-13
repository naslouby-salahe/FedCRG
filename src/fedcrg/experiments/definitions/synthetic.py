"""The locked S1-S6 synthetic/robustness-simulation programme: kernels and orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import binom, gamma, lognorm, norm

from fedcrg.artifacts.json_io import atomic_write_json
from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.domain.enums import (
    ContaminationDirection,
    ExperimentAxisId,
    ExperimentId,
    SyntheticDistribution,
)
from fedcrg.domain.values import OperatingBand
from fedcrg.experiments.experiment_definition import ExperimentDefinition, get_experiment_definition
from fedcrg.method.calibration_readiness import ReadinessPlanBuilder
from fedcrg.method.mismatch_detection import clopper_pearson_interval

# --- Statistical kernels -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntheticCoverageResult:
    experiment: ExperimentId
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


@dataclass(frozen=True, slots=True)
class RobustnessCell:
    axis: ExperimentAxisId
    value: float
    coverage: float
    repetitions: int


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
    experiment: ExperimentId,
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
        experiment=ExperimentId.CALIBRATION_CONTAMINATION,
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


def _threshold(scores: np.ndarray, rank: int) -> float:
    return float(np.sort(scores, kind="stable")[rank - 1])


def temporal_dependence_stress(
    phi: float,
    sample_count: int,
    repetitions: int,
    band: OperatingBand,
    assurance: float,
    seed: int,
) -> RobustnessCell:
    if not -1.0 < phi < 1.0:
        raise ValueError("AR(1) phi must be strictly inside (-1,1)")
    plan = ReadinessPlanBuilder().build(sample_count, band, assurance)
    rng = np.random.Generator(np.random.PCG64(seed))
    inside = 0
    for _ in range(repetitions):
        innovations = rng.normal(size=sample_count)
        series = np.empty(sample_count, dtype=np.float64)
        series[0] = innovations[0]
        scale = np.sqrt(1.0 - phi**2)
        for index in range(1, sample_count):
            series[index] = phi * series[index - 1] + scale * innovations[index]
        fpr = 1.0 - norm.cdf(_threshold(series, plan.rank))
        inside += int(band.lower <= fpr <= band.upper)
    return RobustnessCell(
        ExperimentAxisId.PHI,
        phi,
        inside / repetitions,
        repetitions,
    )


def calibration_shift_stress(
    mean_shift: float,
    repetitions: int,
    band: OperatingBand,
    assurance: float,
    seed: int,
    sample_count: int = 2000,
) -> RobustnessCell:
    plan = ReadinessPlanBuilder().build(sample_count, band, assurance)
    rng = np.random.Generator(np.random.PCG64(seed))
    inside = 0
    for _ in range(repetitions):
        threshold = _threshold(rng.normal(size=sample_count), plan.rank)
        future_fpr = 1.0 - norm.cdf(threshold, loc=mean_shift, scale=1.0)
        inside += int(band.lower <= future_fpr <= band.upper)
    return RobustnessCell(
        ExperimentAxisId.MEAN_SHIFT,
        mean_shift,
        inside / repetitions,
        repetitions,
    )


# --- Programme orchestration --------------------------------------------------------

SyntheticCell = SyntheticCoverageResult | RobustnessCell | MismatchPowerResult


@dataclass(frozen=True, slots=True)
class SyntheticExperimentEnvelope:
    experiment_id: ExperimentId
    expected_monte_carlo_trials: int
    expected_exact_cells: int
    actual_monte_carlo_trials: int
    actual_cells: int
    cells: tuple[SyntheticCell, ...]


class RunSyntheticExperiments:
    @staticmethod
    def _int_values(definition: ExperimentDefinition, axis: ExperimentAxisId) -> tuple[int, ...]:
        values = definition.axis(axis).values
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise TypeError(f"{definition.id.value}/{axis.value} must contain integers")
        return tuple(int(value) for value in values)

    @staticmethod
    def _float_values(
        definition: ExperimentDefinition, axis: ExperimentAxisId
    ) -> tuple[float, ...]:
        values = definition.axis(axis).values
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
        ):
            raise TypeError(f"{definition.id.value}/{axis.value} must contain numbers")
        return tuple(float(value) for value in values)

    @staticmethod
    def _distributions(definition: ExperimentDefinition) -> tuple[SyntheticDistribution, ...]:
        values = definition.axis(ExperimentAxisId.DISTRIBUTION).values
        if not all(isinstance(value, SyntheticDistribution) for value in values):
            raise TypeError(f"{definition.id.value} distribution axis is malformed")
        return tuple(value for value in values if isinstance(value, SyntheticDistribution))

    @staticmethod
    def _directions(definition: ExperimentDefinition) -> tuple[ContaminationDirection, ...]:
        values = definition.axis(ExperimentAxisId.DIRECTION).values
        if not all(isinstance(value, ContaminationDirection) for value in values):
            raise TypeError(f"{definition.id.value} direction axis is malformed")
        return tuple(value for value in values if isinstance(value, ContaminationDirection))

    @staticmethod
    def _write(
        output: Path,
        definition: ExperimentDefinition,
        cells: tuple[SyntheticCell, ...],
        actual_trials: int,
    ) -> Path:
        envelope = SyntheticExperimentEnvelope(
            experiment_id=definition.id,
            expected_monte_carlo_trials=definition.workload.monte_carlo_trials,
            expected_exact_cells=definition.workload.exact_cells,
            actual_monte_carlo_trials=actual_trials,
            actual_cells=len(cells),
            cells=cells,
        )
        if definition.workload.monte_carlo_trials and (
            actual_trials != definition.workload.monte_carlo_trials
        ):
            raise RuntimeError(
                f"{definition.id.value} trial ledger mismatch: "
                f"{actual_trials} != {definition.workload.monte_carlo_trials}"
            )
        if definition.workload.exact_cells and (len(cells) != definition.workload.exact_cells):
            raise RuntimeError(
                f"{definition.id.value} exact-cell ledger mismatch: "
                f"{len(cells)} != {definition.workload.exact_cells}"
            )
        atomic_write_json(output, envelope)
        return output

    def run_s1(self, config: ExperimentConfig, output: Path) -> Path:
        definition = get_experiment_definition(ExperimentId.READINESS_THEOREM)
        repetitions = self._int_values(definition, ExperimentAxisId.REPETITIONS)[0]
        cells = tuple(
            iid_readiness_validation(
                ExperimentId.READINESS_THEOREM,
                distribution,
                sample_count,
                repetitions,
                alpha=config.protocol.alpha,
                rho=config.protocol.rho,
                assurance=config.protocol.readiness_assurance,
                seed=config.randomness.synthetic_seed + sample_count,
            )
            for distribution in self._distributions(definition)
            for sample_count in self._int_values(definition, ExperimentAxisId.CALIBRATION_N)
        )
        return self._write(output, definition, cells, len(cells) * repetitions)

    def run_s2(self, config: ExperimentConfig, output: Path) -> Path:
        definition = get_experiment_definition(ExperimentId.TARGET_FPR_SYNTHETIC)
        repetitions = self._int_values(definition, ExperimentAxisId.REPETITIONS)[0]
        distributions = self._distributions(definition)
        rows: list[SyntheticCoverageResult] = []
        for cell in definition.coupled_cells:
            alpha = float(cell.value(ExperimentAxisId.ALPHA))
            sample_count = int(cell.value(ExperimentAxisId.CALIBRATION_N))
            for distribution in distributions:
                rows.append(
                    iid_readiness_validation(
                        ExperimentId.TARGET_FPR_SYNTHETIC,
                        distribution,
                        sample_count,
                        repetitions,
                        alpha=alpha,
                        rho=config.protocol.rho,
                        assurance=config.protocol.readiness_assurance,
                        seed=config.randomness.synthetic_seed + sample_count,
                    )
                )
        cells = tuple(rows)
        return self._write(output, definition, cells, len(cells) * repetitions)

    def run_s3(self, config: ExperimentConfig, output: Path) -> Path:
        definition = get_experiment_definition(ExperimentId.TEMPORAL_DEPENDENCE)
        repetitions = self._int_values(definition, ExperimentAxisId.REPETITIONS)[0]
        cells = tuple(
            temporal_dependence_stress(
                phi,
                sample_count,
                repetitions,
                config.protocol.band,
                config.protocol.readiness_assurance,
                config.randomness.synthetic_seed + sample_count + int(phi * 1000),
            )
            for phi in self._float_values(definition, ExperimentAxisId.PHI)
            for sample_count in self._int_values(definition, ExperimentAxisId.CALIBRATION_N)
        )
        return self._write(output, definition, cells, len(cells) * repetitions)

    def run_s4(self, config: ExperimentConfig, output: Path) -> Path:
        definition = get_experiment_definition(ExperimentId.CALIBRATION_SHIFT)
        repetitions = self._int_values(definition, ExperimentAxisId.REPETITIONS)[0]
        cells = tuple(
            calibration_shift_stress(
                shift,
                repetitions,
                config.protocol.band,
                config.protocol.readiness_assurance,
                config.randomness.synthetic_seed + int(shift * 1000),
            )
            for shift in self._float_values(definition, ExperimentAxisId.MEAN_SHIFT)
        )
        return self._write(output, definition, cells, len(cells) * repetitions)

    def run_s5(self, config: ExperimentConfig, output: Path) -> Path:
        definition = get_experiment_definition(ExperimentId.CALIBRATION_CONTAMINATION)
        repetitions = self._int_values(definition, ExperimentAxisId.REPETITIONS)[0]
        cells = tuple(
            contamination_validation(
                fraction,
                direction,
                repetitions,
                sample_count=config.dataset.split.calibration_benign,
                seed=config.randomness.synthetic_seed + int(fraction * 1_000_000),
            )
            for fraction in self._float_values(definition, ExperimentAxisId.FRACTION)
            for direction in self._directions(definition)
        )
        return self._write(output, definition, cells, len(cells) * repetitions)

    def run_s6(self, output: Path) -> Path:
        definition = get_experiment_definition(ExperimentId.MISMATCH_POWER)
        cells = tuple(
            exact_mismatch_power(sample_count, true_fpr)
            for sample_count in self._int_values(definition, ExperimentAxisId.MISMATCH_N)
            for true_fpr in self._float_values(definition, ExperimentAxisId.TRUE_FPR)
        )
        return self._write(output, definition, cells, 0)
