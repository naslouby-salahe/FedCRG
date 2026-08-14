"""Statistical analyses: synthetic programme kernels, real-score sensitivities,
confirmatory contrasts, bootstrap intervals, and split-stability summaries.

All analyses read their scientific values from the typed configuration and the
locked experiment catalogue; no kernel hardcodes a protocol constant.
"""

from __future__ import annotations

from collections.abc import Callable
from math import sqrt
from pathlib import Path
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, TypeAdapter
from scipy.stats import binom, gamma, lognorm, norm

from fedcrg.config import ExperimentConfig, ExperimentSpec, ProtocolConfig, SplitConfig
from fedcrg.thresholding.metrics import FederationMetrics
from fedcrg.thresholding.readiness import (
    BinomialCounts,
    CalibrationReadinessEvaluator,
    DeploymentDecision,
    OperatingBand,
    ReadinessPlanBuilder,
    ReferenceMismatchEvaluator,
    build_reference_threshold,
    clopper_pearson_interval,
    familywise_readiness_assurance,
    minimum_bidirectional_sample_count,
)
from fedcrg.types import (
    Alpha,
    AnalysisSeed,
    Assurance,
    BlockCount,
    BootstrapReplicateCount,
    ByteCount,
    CalibrationSeed,
    ClientId,
    ConfidenceLevel,
    ContaminationDirection,
    ContaminationFraction,
    DatasetId,
    DecisionState,
    Duration,
    ExperimentAxisId,
    ExperimentId,
    ExperimentStatus,
    Fpr,
    Fraction,
    Identifier,
    Metric,
    ModelSeed,
    NonNegativeCount,
    PolicyId,
    PositiveCount,
    RepetitionCount,
    RunId,
    SampleCount,
    Score,
    SyntheticDistribution,
    Tpr,
    TrueFpr,
    WarmupCount,
)

Frozen = ConfigDict(frozen=True)

_CLIENT_ID_ADAPTER = TypeAdapter(ClientId)


class BenchmarkPrimitive(StrEnum):
    """Timed protocol primitives of the computational benchmark."""

    REFERENCE_CONSTRUCTION = "reference_construction"
    READINESS_ORDER_STATISTIC = "readiness_order_statistic"
    MISMATCH_EVIDENCE = "mismatch_evidence"
    DEPLOYMENT_DECISION = "deployment_decision"


class SyntheticCoverageResult(BaseModel):
    """Exact and empirical coverage for one synthetic condition."""

    model_config = Frozen

    experiment: ExperimentId
    condition: SyntheticDistribution | ContaminationDirection
    sample_count: SampleCount
    exact_probability: Assurance
    empirical_probability: Fraction
    repetitions: RepetitionCount
    accepted: bool | None = None


class MismatchPowerResult(BaseModel):
    """Exact mismatch-declaration probability for one cell."""

    model_config = Frozen

    sample_count: SampleCount
    true_fpr: Fpr
    declaration_probability: Fraction


class RobustnessCell(BaseModel):
    """One stress-axis coverage cell."""

    model_config = Frozen

    axis: ExperimentAxisId
    value: Metric
    coverage: Fraction
    repetitions: RepetitionCount


SyntheticCell = SyntheticCoverageResult | RobustnessCell | MismatchPowerResult


class SyntheticExperimentEnvelope(BaseModel):
    """Locked ledger and cells of one synthetic experiment."""

    model_config = Frozen

    experiment_id: ExperimentId
    expected_monte_carlo_trials: NonNegativeCount
    expected_exact_cells: NonNegativeCount
    actual_monte_carlo_trials: NonNegativeCount
    actual_cells: NonNegativeCount
    cells: tuple[SyntheticCell, ...]


class PairedBootstrapInterval(BaseModel):
    """Bootstrap interval over paired model-seed indices."""

    model_config = Frozen

    observed_difference: Metric
    lower: Metric
    upper: Metric
    replicates: BootstrapReplicateCount
    seed: AnalysisSeed


class DescriptiveSummary(BaseModel):
    """Descriptive statistics of one value vector."""

    model_config = Frozen

    values: tuple[Metric, ...]
    mean: Metric
    standard_deviation: Metric
    median: Metric
    minimum: Metric
    maximum: Metric


class SplitSensitivitySummary(BaseModel):
    """Percentile summary of repeated calibration splits."""

    model_config = Frozen

    median: Metric
    iqr: Metric
    p05: Metric
    p95: Metric


def describe(values: tuple[Metric, ...]) -> DescriptiveSummary:
    """Descriptive statistics of one finite non-empty vector."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) == 0 or not np.isfinite(data).all():
        raise ValueError("Summary values must be finite and non-empty")
    return DescriptiveSummary(
        values=values,
        mean=float(np.mean(data)),
        standard_deviation=float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        median=float(np.median(data)),
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
    )


def split_sensitivity_summary(values: tuple[Metric, ...]) -> SplitSensitivitySummary:
    """Percentile summary of one split-sensitivity vector."""
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0:
        raise ValueError("Split sensitivity requires values")
    return SplitSensitivitySummary(
        median=float(np.median(data)),
        iqr=float(np.percentile(data, 75) - np.percentile(data, 25)),
        p05=float(np.percentile(data, 5)),
        p95=float(np.percentile(data, 95)),
    )


def paired_model_seed_bootstrap(
    method: tuple[Metric, ...],
    comparator: tuple[Metric, ...],
    *,
    replicates: BootstrapReplicateCount,
    seed: AnalysisSeed,
) -> PairedBootstrapInterval:
    """Bootstrap paired model-seed indices, never treat calibration splits as subjects."""
    left = np.asarray(method, dtype=np.float64)
    right = np.asarray(comparator, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) == 0:
        raise ValueError("Paired bootstrap inputs must be aligned non-empty vectors")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    indices = rng.integers(0, len(left), size=(replicates, len(left)))
    differences = np.mean(left[indices] - right[indices], axis=1)
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return PairedBootstrapInterval(
        observed_difference=float(np.mean(left - right)),
        lower=float(lower),
        upper=float(upper),
        replicates=replicates,
        seed=seed,
    )


def draw_distribution(
    rng: np.random.Generator,
    distribution: SyntheticDistribution,
    size: SampleCount,
) -> np.ndarray:
    """Draw one synthetic sample from a locked distribution."""
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
        values[component] = rng.normal(loc=3.0, scale=1.0, size=int(component.sum()))
        return values
    raise AssertionError(f"Unhandled synthetic distribution: {distribution}")


def distribution_cdf(distribution: SyntheticDistribution, threshold: Score) -> Metric:
    """CDF of one synthetic distribution at a threshold."""
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
    sample_count: SampleCount,
    repetitions: RepetitionCount,
    *,
    alpha: Alpha,
    rho: Fraction,
    assurance: Assurance,
    seed: AnalysisSeed,
) -> SyntheticCoverageResult:
    """Empirical readiness coverage for IID synthetic scores."""
    band = OperatingBand(
        lower=max(0.0, alpha * (1.0 - rho)),
        upper=min(1.0, alpha * (1.0 + rho)),
    )
    plan = ReadinessPlanBuilder().build(sample_count, band, assurance)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    inside = 0
    for _ in range(repetitions):
        values = np.sort(draw_distribution(rng, distribution, sample_count), kind="stable")
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
    fraction: ContaminationFraction,
    direction: ContaminationDirection,
    repetitions: RepetitionCount,
    *,
    sample_count: SampleCount,
    seed: AnalysisSeed,
    band: OperatingBand,
    assurance: Assurance,
) -> SyntheticCoverageResult:
    """Empirical readiness coverage under score contamination."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Contamination fraction must be in [0,1]")
    plan = ReadinessPlanBuilder().build(sample_count, band, assurance)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    inside = 0
    contamination_count = int(round(fraction * sample_count))
    location = 3.0 if direction is ContaminationDirection.HIGH else -3.0
    for _ in range(repetitions):
        values = rng.normal(size=sample_count)
        if contamination_count:
            indices = rng.choice(sample_count, size=contamination_count, replace=False)
            values[indices] = rng.normal(loc=location, scale=1.0, size=contamination_count)
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
    sample_count: SampleCount,
    true_fpr: TrueFpr,
    band: OperatingBand,
    confidence: ConfidenceLevel,
) -> MismatchPowerResult:
    """Exact mismatch-declaration probability from binomial tails."""
    if sample_count <= 0 or not 0.0 <= true_fpr <= 1.0:
        raise ValueError("Mismatch power requires n>0 and true_fpr in [0,1]")
    low_counts: list[NonNegativeCount] = []
    high_counts: list[NonNegativeCount] = []
    for exceedances in range(sample_count + 1):
        interval = clopper_pearson_interval(BinomialCounts(exceedances, sample_count), confidence)
        if interval.upper < band.lower:
            low_counts.append(exceedances)
        elif interval.lower > band.upper:
            high_counts.append(exceedances)
    probability = 0.0
    if low_counts:
        probability += float(binom.cdf(max(low_counts), sample_count, true_fpr))
    if high_counts:
        probability += float(binom.sf(min(high_counts) - 1, sample_count, true_fpr))
    return MismatchPowerResult(
        sample_count=sample_count, true_fpr=true_fpr, declaration_probability=probability
    )


def _threshold(scores: np.ndarray, rank: PositiveCount) -> Score:
    return float(np.sort(scores, kind="stable")[rank - 1])


def temporal_dependence_stress(
    phi: Metric,
    sample_count: SampleCount,
    repetitions: RepetitionCount,
    band: OperatingBand,
    assurance: Assurance,
    seed: AnalysisSeed,
) -> RobustnessCell:
    """Readiness coverage under AR(1) temporal dependence."""
    if not -1.0 < phi < 1.0:
        raise ValueError("AR(1) phi must be strictly inside (-1,1)")
    plan = ReadinessPlanBuilder().build(sample_count, band, assurance)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
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
        axis=ExperimentAxisId.PHI,
        value=phi,
        coverage=inside / repetitions,
        repetitions=repetitions,
    )


def calibration_shift_stress(
    mean_shift: Metric,
    repetitions: RepetitionCount,
    band: OperatingBand,
    assurance: Assurance,
    seed: AnalysisSeed,
    sample_count: SampleCount,
) -> RobustnessCell:
    """Readiness coverage under a calibration mean shift."""
    plan = ReadinessPlanBuilder().build(sample_count, band, assurance)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    inside = 0
    for _ in range(repetitions):
        threshold = _threshold(rng.normal(size=sample_count), plan.rank)
        future_fpr = 1.0 - norm.cdf(threshold, loc=mean_shift, scale=1.0)
        inside += int(band.lower <= future_fpr <= band.upper)
    return RobustnessCell(
        axis=ExperimentAxisId.MEAN_SHIFT,
        value=mean_shift,
        coverage=inside / repetitions,
        repetitions=repetitions,
    )


class RunSyntheticExperiments:
    """Execute the locked synthetic/robustness programme from catalogue axes."""

    @staticmethod
    def _int_values(spec: ExperimentSpec, axis: ExperimentAxisId) -> tuple[NonNegativeCount, ...]:
        values = spec.axis(axis).values
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise TypeError(f"{spec.id.value}/{axis.value} must contain integers")
        return tuple(int(value) for value in values)

    @staticmethod
    def _float_values(spec: ExperimentSpec, axis: ExperimentAxisId) -> tuple[Metric, ...]:
        values = spec.axis(axis).values
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
        ):
            raise TypeError(f"{spec.id.value}/{axis.value} must contain numbers")
        return tuple(float(value) for value in values)

    @staticmethod
    def _distributions(spec: ExperimentSpec) -> tuple[SyntheticDistribution, ...]:
        values = spec.axis(ExperimentAxisId.DISTRIBUTION).values
        if not all(isinstance(value, SyntheticDistribution) for value in values):
            raise TypeError(f"{spec.id.value} distribution axis is malformed")
        return tuple(value for value in values if isinstance(value, SyntheticDistribution))

    @staticmethod
    def _directions(spec: ExperimentSpec) -> tuple[ContaminationDirection, ...]:
        values = spec.axis(ExperimentAxisId.DIRECTION).values
        if not all(isinstance(value, ContaminationDirection) for value in values):
            raise TypeError(f"{spec.id.value} direction axis is malformed")
        return tuple(value for value in values if isinstance(value, ContaminationDirection))

    @staticmethod
    def write(
        output: Path,
        spec: ExperimentSpec,
        cells: tuple[SyntheticCell, ...],
        actual_trials: NonNegativeCount,
    ) -> Path:
        envelope = SyntheticExperimentEnvelope(
            experiment_id=spec.id,
            expected_monte_carlo_trials=spec.workload.monte_carlo_trials,
            expected_exact_cells=spec.workload.exact_cells,
            actual_monte_carlo_trials=actual_trials,
            actual_cells=len(cells),
            cells=cells,
        )
        if spec.workload.monte_carlo_trials and actual_trials != spec.workload.monte_carlo_trials:
            raise RuntimeError(
                f"{spec.id.value} trial ledger mismatch: "
                f"{actual_trials} != {spec.workload.monte_carlo_trials}"
            )
        if spec.workload.exact_cells and len(cells) != spec.workload.exact_cells:
            raise RuntimeError(
                f"{spec.id.value} exact-cell ledger mismatch: "
                f"{len(cells)} != {spec.workload.exact_cells}"
            )
        from fedcrg.evidence.store import atomic_write_json

        atomic_write_json(output, envelope)
        return output

    def run(
        self,
        experiment_id: ExperimentId,
        spec: ExperimentSpec,
        config: ExperimentConfig,
        output: Path,
    ) -> Path:
        try:
            runner = self._RUNNERS[experiment_id]
        except KeyError:
            raise ValueError(f"Unsupported synthetic experiment: {experiment_id.value}") from None
        return runner(self, spec, config, output)

    def _run_readiness_theorem(
        self, spec: ExperimentSpec, config: ExperimentConfig, output: Path
    ) -> Path:
        repetitions = self._int_values(spec, ExperimentAxisId.REPETITIONS)[0]
        cells = tuple(
            iid_readiness_validation(
                ExperimentId.READINESS_THEOREM,
                distribution,
                sample_count,
                repetitions,
                alpha=config.protocol.alpha,
                rho=config.protocol.rho,
                assurance=config.protocol.readiness_assurance,
                seed=int(config.randomness.synthetic_seed + sample_count),
            )
            for distribution in self._distributions(spec)
            for sample_count in self._int_values(spec, ExperimentAxisId.CALIBRATION_N)
        )
        return self.write(output, spec, cells, len(cells) * repetitions)

    def _run_target_fpr_synthetic(
        self, spec: ExperimentSpec, config: ExperimentConfig, output: Path
    ) -> Path:
        repetitions = self._int_values(spec, ExperimentAxisId.REPETITIONS)[0]
        rows: list[SyntheticCoverageResult] = []
        for cell in spec.coupled_cells:
            alpha = float(cell.value(ExperimentAxisId.ALPHA))
            sample_count = int(cell.value(ExperimentAxisId.CALIBRATION_N))
            for distribution in self._distributions(spec):
                rows.append(
                    iid_readiness_validation(
                        ExperimentId.TARGET_FPR_SYNTHETIC,
                        distribution,
                        sample_count,
                        repetitions,
                        alpha=alpha,
                        rho=config.protocol.rho,
                        assurance=config.protocol.readiness_assurance,
                        seed=int(config.randomness.synthetic_seed + sample_count),
                    )
                )
        return self.write(output, spec, tuple(rows), len(rows) * repetitions)

    def _run_temporal_dependence(
        self, spec: ExperimentSpec, config: ExperimentConfig, output: Path
    ) -> Path:
        repetitions = self._int_values(spec, ExperimentAxisId.REPETITIONS)[0]
        cells = tuple(
            temporal_dependence_stress(
                phi,
                sample_count,
                repetitions,
                config.protocol.band,
                config.protocol.readiness_assurance,
                int(config.randomness.synthetic_seed + sample_count + int(phi * 1000)),
            )
            for phi in self._float_values(spec, ExperimentAxisId.PHI)
            for sample_count in self._int_values(spec, ExperimentAxisId.CALIBRATION_N)
        )
        return self.write(output, spec, cells, len(cells) * repetitions)

    def _run_calibration_shift(
        self, spec: ExperimentSpec, config: ExperimentConfig, output: Path
    ) -> Path:
        repetitions = self._int_values(spec, ExperimentAxisId.REPETITIONS)[0]
        cells = tuple(
            calibration_shift_stress(
                shift,
                repetitions,
                config.protocol.band,
                config.protocol.readiness_assurance,
                int(config.randomness.synthetic_seed + int(shift * 1000)),
                sample_count=config.dataset.split.calibration_benign,
            )
            for shift in self._float_values(spec, ExperimentAxisId.MEAN_SHIFT)
        )
        return self.write(output, spec, cells, len(cells) * repetitions)

    def _run_calibration_contamination(
        self, spec: ExperimentSpec, config: ExperimentConfig, output: Path
    ) -> Path:
        repetitions = self._int_values(spec, ExperimentAxisId.REPETITIONS)[0]
        cells = tuple(
            contamination_validation(
                fraction,
                direction,
                repetitions,
                sample_count=config.dataset.split.calibration_benign,
                seed=int(config.randomness.synthetic_seed + int(fraction * 1000)),
                band=config.protocol.band,
                assurance=config.protocol.readiness_assurance,
            )
            for fraction in self._float_values(spec, ExperimentAxisId.FRACTION)
            for direction in self._directions(spec)
        )
        return self.write(output, spec, cells, len(cells) * repetitions)

    def _run_mismatch_power(
        self, spec: ExperimentSpec, config: ExperimentConfig, output: Path
    ) -> Path:
        cells = tuple(
            exact_mismatch_power(
                sample_count,
                true_fpr,
                band=config.protocol.band,
                confidence=config.protocol.mismatch_confidence,
            )
            for sample_count in self._int_values(spec, ExperimentAxisId.MISMATCH_N)
            for true_fpr in self._float_values(spec, ExperimentAxisId.TRUE_FPR)
        )
        return self.write(output, spec, cells, 0)

    _RUNNERS: dict[
        ExperimentId,
        Callable[[RunSyntheticExperiments, ExperimentSpec, ExperimentConfig, Path], Path],
    ] = {
        ExperimentId.READINESS_THEOREM: _run_readiness_theorem,
        ExperimentId.TARGET_FPR_SYNTHETIC: _run_target_fpr_synthetic,
        ExperimentId.TEMPORAL_DEPENDENCE: _run_temporal_dependence,
        ExperimentId.CALIBRATION_SHIFT: _run_calibration_shift,
        ExperimentId.CALIBRATION_CONTAMINATION: _run_calibration_contamination,
        ExperimentId.MISMATCH_POWER: _run_mismatch_power,
    }


class FederationResultRecord(BaseModel):
    """One completed run's federation-endpoint record."""

    model_config = Frozen

    run_id: RunId
    experiment_id: ExperimentId
    dataset_id: DatasetId
    model_seed: ModelSeed
    calibration_seed: CalibrationSeed
    policy: PolicyId
    mebe: Metric
    high_excess: Metric
    band_violation_rate: Fraction
    attack_balanced_macro_tpr: Tpr | None


def load_federation_results(run_dirs: tuple[Path, ...]) -> tuple[FederationResultRecord, ...]:
    """Load completed federation records from run directories."""
    from fedcrg.evidence.models import RunConfig, RunManifest
    from fedcrg.evidence.store import RunLayout, load_json_model

    rows: list[FederationResultRecord] = []
    for run_dir in run_dirs:
        layout = RunLayout(run_dir)
        if (
            not layout.manifest.is_file()
            or not layout.federation_metrics.is_file()
            or not layout.run_config.is_file()
        ):
            continue
        try:
            manifest = RunManifest.model_validate_json(layout.manifest.read_text(encoding="utf-8"))
            federation = load_json_model(layout.federation_metrics, FederationMetrics)
            config_payload = RunConfig.model_validate_json(
                layout.run_config.read_text(encoding="utf-8")
            )
            dataset_id = config_payload.parameters.dataset.id
        except Exception:
            continue
        if manifest.status is not ExperimentStatus.COMPLETE:
            continue
        rows.append(
            FederationResultRecord(
                run_id=manifest.run_id,
                experiment_id=manifest.experiment_id,
                dataset_id=dataset_id,
                model_seed=manifest.model_seed,
                calibration_seed=manifest.calibration_seed,
                policy=manifest.policy_id,
                mebe=federation.mebe,
                high_excess=federation.high_excess,
                band_violation_rate=federation.band_violation_rate,
                attack_balanced_macro_tpr=federation.attack_balanced_macro_tpr,
            )
        )
    return tuple(rows)


class ContrastMetricResult(BaseModel):
    """One confirmatory contrast metric result."""

    model_config = Frozen

    metric: Identifier
    method_summary: DescriptiveSummary
    comparator_summary: DescriptiveSummary
    paired_difference: PairedBootstrapInterval
    relative_difference: Metric | None = None


class PolicyContrastResult(BaseModel):
    """All contrast metrics against one comparator policy."""

    model_config = Frozen

    comparator: PolicyId
    metrics: tuple[ContrastMetricResult, ...]


def confirmatory_contrasts(
    records: tuple[FederationResultRecord, ...],
    *,
    named_calibration_seed: CalibrationSeed,
    expected_model_seeds: tuple[ModelSeed, ...],
    bootstrap_seed: AnalysisSeed,
    bootstrap_replicates: BootstrapReplicateCount,
) -> tuple[PolicyContrastResult, ...]:
    """Compute exactly the four pre-registered method-minus-comparator contrasts."""
    comparators = (
        PolicyId.GLOBAL_QUANTILE,
        PolicyId.LOCAL_QUANTILE,
        PolicyId.READINESS_ONLY,
        PolicyId.SHRINKAGE,
    )
    selected = tuple(row for row in records if row.calibration_seed == named_calibration_seed)
    expected_seeds = set(expected_model_seeds)
    observed_method_seeds = {row.model_seed for row in selected if row.policy is PolicyId.FEDCRG}
    if observed_method_seeds != expected_seeds:
        raise ValueError(
            "Confirmatory contrasts require exactly the configured model seeds "
            f"for FedCRG, observed {sorted(observed_method_seeds)}"
        )
    method_by_seed = {row.model_seed: row for row in selected if row.policy is PolicyId.FEDCRG}
    results: list[PolicyContrastResult] = []
    for comparator in comparators:
        comparator_by_seed = {row.model_seed: row for row in selected if row.policy is comparator}
        if set(comparator_by_seed) != expected_seeds:
            raise ValueError(
                f"Confirmatory comparator {comparator.value} is missing paired model seeds"
            )
        common_seeds = tuple(sorted(expected_seeds))
        metrics: list[ContrastMetricResult] = []
        for metric in ("mebe", "high_excess", "attack_balanced_macro_tpr"):
            left_values = tuple(getattr(method_by_seed[seed], metric) for seed in common_seeds)
            right_values = tuple(getattr(comparator_by_seed[seed], metric) for seed in common_seeds)
            if any(value is None for value in left_values + right_values):
                continue
            left = tuple(float(value) for value in left_values)
            right = tuple(float(value) for value in right_values)
            bootstrap = paired_model_seed_bootstrap(
                left,
                right,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            comparator_mean = describe(right).mean
            relative = (
                bootstrap.observed_difference / comparator_mean if comparator_mean > 0.0 else None
            )
            metrics.append(
                ContrastMetricResult(
                    metric=metric,
                    method_summary=describe(left),
                    comparator_summary=describe(right),
                    paired_difference=bootstrap,
                    relative_difference=relative,
                )
            )
        results.append(PolicyContrastResult(comparator=comparator, metrics=tuple(metrics)))
    return tuple(results)


class ThresholdStability(BaseModel):
    """Stability summary of one threshold vector."""

    model_config = Frozen

    count: PositiveCount
    standard_deviation: Metric
    iqr: Metric
    minimum: Metric
    maximum: Metric


class StateFrequency(BaseModel):
    """Observed frequency of one decision state."""

    model_config = Frozen

    state: DecisionState
    frequency: Fraction


class StateStability(BaseModel):
    """Transition and state-frequency summary."""

    model_config = Frozen

    count: PositiveCount
    transition_count: NonNegativeCount
    transition_frequency: Fraction
    state_frequencies: tuple[StateFrequency, ...]


def summarize_threshold_stability(values: tuple[Metric, ...]) -> ThresholdStability:
    """Stability statistics of one threshold vector."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) == 0 or not np.isfinite(data).all():
        raise ValueError("Threshold stability requires finite non-empty values")
    return ThresholdStability(
        count=len(data),
        standard_deviation=float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        iqr=float(np.percentile(data, 75) - np.percentile(data, 25)),
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
    )


def summarize_state_stability(states: tuple[DecisionState, ...]) -> StateStability:
    """Transition and frequency summary of one state sequence."""
    if not states:
        raise ValueError("State stability requires at least one state")
    transitions = sum(
        left is not right for left, right in zip(states[:-1], states[1:], strict=True)
    )
    counts = {state: states.count(state) for state in DecisionState}
    total = len(states)
    return StateStability(
        count=total,
        transition_count=transitions,
        transition_frequency=transitions / max(1, total - 1),
        state_frequencies=tuple(
            StateFrequency(state=state, frequency=count / total) for state, count in counts.items()
        ),
    )


class SplitSensitivityRow(BaseModel):
    """Percentile summary for one model/policy/metric group."""

    model_config = Frozen

    model_seed: ModelSeed
    policy: PolicyId
    metric: Identifier
    median: Metric
    iqr: Metric
    p05: Metric
    p95: Metric
    calibration_split_count: PositiveCount


class StabilityMetricId(StrEnum):
    """Closed set of federation-level stability metrics summarized over splits."""

    MEBE = "mebe"
    HIGH_EXCESS = "high_excess"
    ATTACK_BALANCED_MACRO_TPR = "attack_balanced_macro_tpr"


def split_sensitivity(
    records: tuple[FederationResultRecord, ...],
) -> tuple[SplitSensitivityRow, ...]:
    """Summarize repeated role permutations without treating them as independent subjects."""
    grouped: dict[tuple[ModelSeed, PolicyId, StabilityMetricId], list[Metric]] = {}
    for row in records:
        for metric in StabilityMetricId:
            value = getattr(row, metric.value)
            if value is None:
                continue
            grouped.setdefault((row.model_seed, row.policy, metric), []).append(float(value))
    rows: list[SplitSensitivityRow] = []
    for (model_seed, policy, metric), values in sorted(
        grouped.items(),
        key=lambda item: (int(item[0][0]), item[0][1].value, item[0][2].value),
    ):
        summary = split_sensitivity_summary(tuple(values))
        rows.append(
            SplitSensitivityRow(
                model_seed=model_seed,
                policy=policy,
                metric=metric,
                median=summary.median,
                iqr=summary.iqr,
                p05=summary.p05,
                p95=summary.p95,
                calibration_split_count=len(values),
            )
        )
    return tuple(rows)


def source_order_blocks(scores: np.ndarray, block_count: BlockCount) -> tuple[np.ndarray, ...]:
    """Split one score vector into ordered source blocks."""
    values = np.asarray(scores, dtype=np.float64)
    if block_count <= 0 or len(values) < block_count:
        raise ValueError("Source-order block analysis needs at least one row per block")
    return tuple(
        np.asarray(block, dtype=np.float64) for block in np.array_split(values, block_count)
    )


def contaminate_benign_scores(
    benign: np.ndarray,
    attack_dev: np.ndarray,
    fraction: ContaminationFraction,
    seed: AnalysisSeed,
) -> np.ndarray:
    """Inject attack-development scores into benign scores."""
    values = np.asarray(benign, dtype=np.float64).copy()
    attacks = np.asarray(attack_dev, dtype=np.float64)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Contamination fraction must be in [0,1]")
    count = int(round(fraction * len(values)))
    if count == 0:
        return values
    if count > len(attacks):
        raise ValueError("Attack-development cache is too small for requested contamination")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    target = rng.choice(len(values), size=count, replace=False)
    source = rng.choice(len(attacks), size=count, replace=False)
    values[target] = attacks[source]
    return values


class MismatchCutoffCell(BaseModel):
    """Exact low/high exceedance cutoffs for one sample count."""

    model_config = Frozen

    sample_count: SampleCount
    low_max_exceedances: NonNegativeCount | None = None
    high_min_exceedances: NonNegativeCount | None = None


class ProtocolTablePrecomputer:
    """Freeze every readiness/cutoff entry needed by the registered programme."""

    def precompute(
        self,
        config: ExperimentConfig,
        spec: ExperimentSpec,
        root: Path | None = None,
    ) -> tuple[Path, Path]:
        from fedcrg.evidence.store import OutputsLayout, atomic_write_json
        from fedcrg.thresholding.readiness import ReadinessPlanCache

        target_root = root or OutputsLayout(config.outputs_root).cache_analysis
        target_root.mkdir(parents=True, exist_ok=True)
        layout = OutputsLayout(config.outputs_root)
        readiness_path = layout.readiness_plans_file
        mismatch_path = layout.mismatch_cutoffs_file
        cache = ReadinessPlanCache(readiness_path)
        for sample_count, protocol in self._readiness_cells(config, spec):
            cache.precompute(
                sample_count,
                protocol.band,
                protocol.readiness_assurance,
            )
        cache.save()

        if self._has_axis(spec, ExperimentAxisId.MISMATCH_N):
            mismatch_counts = tuple(
                int(value) for value in spec.axis(ExperimentAxisId.MISMATCH_N).values
            )
            mismatch_rows = tuple(
                self._mismatch_cutoffs(
                    sample_count,
                    config.protocol.band,
                    config.protocol.mismatch_confidence,
                )
                for sample_count in mismatch_counts
            )
            atomic_write_json(
                mismatch_path,
                {
                    "band": config.protocol.band,
                    "confidence": config.protocol.mismatch_confidence,
                    "minimum_bidirectional_sample_count": minimum_bidirectional_sample_count(
                        config.protocol.band.lower,
                        config.protocol.mismatch_confidence,
                    ),
                    "cells": mismatch_rows,
                },
            )
        return readiness_path, mismatch_path

    @staticmethod
    def _has_axis(spec: ExperimentSpec, axis_id: ExperimentAxisId) -> bool:
        return any(item.id is axis_id for item in spec.axes)

    def _readiness_cells(
        self, config: ExperimentConfig, spec: ExperimentSpec
    ) -> tuple[tuple[SampleCount, ProtocolConfig], ...]:
        cells: dict[tuple[SampleCount, Alpha, Fraction, Assurance], ProtocolConfig] = {}

        def add(
            sample_count: SampleCount,
            *,
            alpha: Alpha,
            rho: Fraction,
            assurance: Assurance,
        ) -> None:
            protocol = ProtocolConfig(
                id=config.protocol.id,
                version=config.protocol.version,
                alpha=alpha,
                rho=rho,
                readiness_assurance=assurance,
                mismatch_confidence=config.protocol.mismatch_confidence,
                strict_exceedance=config.protocol.strict_exceedance,
                reject_calibration_ties=config.protocol.reject_calibration_ties,
            )
            cells[(sample_count, alpha, rho, assurance)] = protocol

        primary_n = config.dataset.split.calibration_benign
        add(
            primary_n,
            alpha=config.protocol.alpha,
            rho=config.protocol.rho,
            assurance=config.protocol.readiness_assurance,
        )
        if self._has_axis(spec, ExperimentAxisId.CALIBRATION_N):
            for value in spec.axis(ExperimentAxisId.CALIBRATION_N).values:
                add(
                    int(value),
                    alpha=config.protocol.alpha,
                    rho=config.protocol.rho,
                    assurance=config.protocol.readiness_assurance,
                )
        if self._has_axis(spec, ExperimentAxisId.READINESS_ASSURANCE):
            for value in spec.axis(ExperimentAxisId.READINESS_ASSURANCE).values:
                add(
                    primary_n,
                    alpha=config.protocol.alpha,
                    rho=config.protocol.rho,
                    assurance=float(value),
                )
        add(
            primary_n,
            alpha=config.protocol.alpha,
            rho=config.protocol.rho,
            assurance=familywise_readiness_assurance(
                config.dataset.expected_clients or config.dataset.minimum_clients,
                config.statistics.familywise_alpha,
            ),
        )
        for cell in spec.coupled_cells:
            add(
                int(cell.value(ExperimentAxisId.CALIBRATION_N)),
                alpha=float(cell.value(ExperimentAxisId.ALPHA)),
                rho=config.protocol.rho,
                assurance=config.protocol.readiness_assurance,
            )
        return tuple((key[0], protocol) for key, protocol in sorted(cells.items()))

    @staticmethod
    def _mismatch_cutoffs(
        sample_count: SampleCount,
        band: OperatingBand,
        confidence: ConfidenceLevel,
    ) -> MismatchCutoffCell:
        lows: list[NonNegativeCount] = []
        highs: list[NonNegativeCount] = []
        for exceedances in range(sample_count + 1):
            interval = clopper_pearson_interval(
                BinomialCounts(exceedances, sample_count), confidence
            )
            if band.lower > 0.0 and interval.upper < band.lower:
                lows.append(exceedances)
            if interval.lower > band.upper:
                highs.append(exceedances)
        return MismatchCutoffCell(
            sample_count=sample_count,
            low_max_exceedances=max(lows) if lows else None,
            high_min_exceedances=min(highs) if highs else None,
        )


class BenchmarkCell(BaseModel):
    """One timed primitive cell of the computational benchmark."""

    model_config = Frozen

    primitive: Identifier
    sample_count: SampleCount
    warmups: WarmupCount
    repetitions: RepetitionCount
    median_seconds: Duration
    p95_seconds: Duration
    peak_rss_bytes: ByteCount


class BenchmarkReport(BaseModel):
    """Environment pin and timed primitive cells."""

    model_config = Frozen

    experiment_id: ExperimentId
    environment: Identifier
    cells: tuple[BenchmarkCell, ...]


class RunBenchmark:
    """Measure the locked runtime primitives on one CPU thread.

    The process is pinned to a single CPU when the operating system allows
    it, warmed up, and then timed for the configured repetition count. Wall
    time is measured with ``time.perf_counter`` and peak resident memory with
    ``resource.getrusage``; medians and 95th percentiles are reported.
    """

    def __init__(self, spec: ExperimentSpec, config: ExperimentConfig) -> None:
        self.spec = spec
        self.config = config

    def _warmups(self) -> WarmupCount:
        return int(self.spec.axis(ExperimentAxisId.WARMUPS).values[0])

    def _repetitions(self) -> RepetitionCount:
        return int(self.spec.axis(ExperimentAxisId.REPETITIONS).values[0])

    def _environment_pin(self) -> Identifier:
        import platform
        import sys

        import numpy as np
        import scipy

        return (
            f"{platform.system()}_{platform.processor()}_python{sys.version.split()[0]}_"
            f"numpy{np.__version__}_scipy{scipy.__version__}"
        )

    @staticmethod
    def _pin_single_cpu() -> None:
        try:
            import os

            os.sched_setaffinity(0, {0})
        except (AttributeError, OSError):
            pass

    def run(self, output: Path) -> Path:
        self._pin_single_cpu()
        warmups = self._warmups()
        repetitions = self._repetitions()
        split = self.config.dataset.split
        client_count = self.config.dataset.expected_clients or self.config.dataset.minimum_clients
        rng = np.random.default_rng(int(self.config.randomness.synthetic_seed))
        reference_scores = {
            _CLIENT_ID_ADAPTER.validate_python(f"c{index:02d}"): rng.normal(
                size=split.reference_benign
            )
            for index in range(client_count)
        }
        calibration_scores = rng.normal(size=split.calibration_benign)
        mismatch_scores = rng.normal(size=split.mismatch_benign)

        plan = ReadinessPlanBuilder().build(
            split.calibration_benign,
            self.config.protocol.band,
            self.config.protocol.readiness_assurance,
        )
        readiness_evaluator = CalibrationReadinessEvaluator()
        mismatch_evaluator = ReferenceMismatchEvaluator()
        decision_engine = DeploymentDecision()
        reference = build_reference_threshold(reference_scores, self.config.protocol.alpha)
        readiness = readiness_evaluator.evaluate(calibration_scores, plan)
        mismatch = mismatch_evaluator.evaluate(
            mismatch_scores,
            reference.value,
            self.config.protocol.band,
            self.config.protocol.mismatch_confidence,
        )

        primitives = (
            (
                BenchmarkPrimitive.REFERENCE_CONSTRUCTION,
                lambda: build_reference_threshold(reference_scores, self.config.protocol.alpha),
            ),
            (
                BenchmarkPrimitive.READINESS_ORDER_STATISTIC,
                lambda: readiness_evaluator.evaluate(calibration_scores, plan),
            ),
            (
                BenchmarkPrimitive.MISMATCH_EVIDENCE,
                lambda: mismatch_evaluator.evaluate(
                    mismatch_scores,
                    reference.value,
                    self.config.protocol.band,
                    self.config.protocol.mismatch_confidence,
                ),
            ),
            (
                BenchmarkPrimitive.DEPLOYMENT_DECISION,
                lambda: decision_engine.decide(
                    reference, readiness, mismatch, self.config.protocol.reject_calibration_ties
                ),
            ),
        )

        cells: list[BenchmarkCell] = []
        for primitive, run_primitive in primitives:
            sample_count = self._sample_count(primitive, split)
            for _ in range(warmups):
                run_primitive()
            timings = _timed_repetitions(run_primitive, repetitions)
            cells.append(
                BenchmarkCell(
                    primitive=primitive.value,
                    sample_count=sample_count,
                    warmups=warmups,
                    repetitions=repetitions,
                    median_seconds=timings.median_seconds,
                    p95_seconds=timings.p95_seconds,
                    peak_rss_bytes=timings.peak_rss_bytes,
                )
            )
        report = BenchmarkReport(
            experiment_id=self.spec.id,
            environment=self._environment_pin(),
            cells=tuple(cells),
        )
        from fedcrg.evidence.store import atomic_write_json

        atomic_write_json(output, report)
        return output

    @staticmethod
    def _sample_count(primitive: BenchmarkPrimitive, split: SplitConfig) -> SampleCount:
        mapping: dict[BenchmarkPrimitive, SampleCount] = {
            BenchmarkPrimitive.REFERENCE_CONSTRUCTION: split.reference_benign,
            BenchmarkPrimitive.READINESS_ORDER_STATISTIC: split.calibration_benign,
            BenchmarkPrimitive.MISMATCH_EVIDENCE: split.mismatch_benign,
            BenchmarkPrimitive.DEPLOYMENT_DECISION: split.calibration_benign,
        }
        return int(mapping[primitive])


class TimedRepetitions(BaseModel):
    """Wall-time percentiles and peak RSS of one timed primitive."""

    model_config = Frozen

    median_seconds: Duration
    p95_seconds: Duration
    peak_rss_bytes: ByteCount


def _timed_repetitions(
    primitive: Callable[[], BaseModel],
    repetitions: RepetitionCount,
) -> TimedRepetitions:
    import resource
    import time

    timings: list[Duration] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        primitive()
        timings.append(time.perf_counter() - started)
    values = np.asarray(timings, dtype=np.float64)
    peak_rss_kilobytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return TimedRepetitions(
        median_seconds=float(np.median(values)),
        p95_seconds=float(np.percentile(values, 95)),
        peak_rss_bytes=int(peak_rss_kilobytes) * 1024,
    )
