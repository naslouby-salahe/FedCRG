"""Application service for the locked S1-S6 synthetic programme."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fedcrg.analysis.robustness import (
    RobustnessCell,
    calibration_shift_stress,
    temporal_dependence_stress,
)
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import (
    ContaminationDirection,
    ExperimentAxisId,
    ExperimentCode,
    ExperimentId,
    SyntheticDistribution,
)
from fedcrg.experiments.models import ExperimentDefinition
from fedcrg.experiments.registry import ExperimentRegistry
from fedcrg.experiments.synthetic import (
    MismatchPowerResult,
    SyntheticCoverageResult,
    contamination_validation,
    exact_mismatch_power,
    iid_readiness_validation,
)

SyntheticCell = SyntheticCoverageResult | RobustnessCell | MismatchPowerResult


@dataclass(frozen=True, slots=True)
class SyntheticExperimentEnvelope:
    experiment_id: ExperimentId
    protocol_code: ExperimentCode
    expected_monte_carlo_trials: int
    expected_exact_cells: int
    actual_monte_carlo_trials: int
    actual_cells: int
    cells: tuple[SyntheticCell, ...]


class RunSyntheticExperiments:
    def __init__(self, registry: ExperimentRegistry | None = None) -> None:
        self.registry = registry or ExperimentRegistry()

    @staticmethod
    def _int_values(definition: ExperimentDefinition, axis: ExperimentAxisId) -> tuple[int, ...]:
        values = definition.axis(axis).values
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            raise TypeError(f"{definition.protocol_code.value}/{axis.value} must contain integers")
        return tuple(int(value) for value in values)

    @staticmethod
    def _float_values(definition: ExperimentDefinition, axis: ExperimentAxisId) -> tuple[float, ...]:
        values = definition.axis(axis).values
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            raise TypeError(f"{definition.protocol_code.value}/{axis.value} must contain numbers")
        return tuple(float(value) for value in values)

    @staticmethod
    def _distributions(definition: ExperimentDefinition) -> tuple[SyntheticDistribution, ...]:
        values = definition.axis(ExperimentAxisId.DISTRIBUTION).values
        if not all(isinstance(value, SyntheticDistribution) for value in values):
            raise TypeError(f"{definition.protocol_code.value} distribution axis is malformed")
        return tuple(value for value in values if isinstance(value, SyntheticDistribution))

    @staticmethod
    def _directions(definition: ExperimentDefinition) -> tuple[ContaminationDirection, ...]:
        values = definition.axis(ExperimentAxisId.DIRECTION).values
        if not all(isinstance(value, ContaminationDirection) for value in values):
            raise TypeError(f"{definition.protocol_code.value} direction axis is malformed")
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
            protocol_code=definition.protocol_code,
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
                f"{definition.protocol_code.value} trial ledger mismatch: "
                f"{actual_trials} != {definition.workload.monte_carlo_trials}"
            )
        if definition.workload.exact_cells and (
            len(cells) != definition.workload.exact_cells
        ):
            raise RuntimeError(
                f"{definition.protocol_code.value} exact-cell ledger mismatch: "
                f"{len(cells)} != {definition.workload.exact_cells}"
            )
        atomic_write_json(output, envelope)
        return output

    def run_s1(self, config: ExperimentConfig, output: Path) -> Path:
        definition = self.registry.get(ExperimentId.READINESS_THEOREM)
        repetitions = self._int_values(definition, ExperimentAxisId.REPETITIONS)[0]
        cells = tuple(
            iid_readiness_validation(
                ExperimentCode.S1,
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
        definition = self.registry.get(ExperimentId.TARGET_FPR_SYNTHETIC)
        repetitions = self._int_values(definition, ExperimentAxisId.REPETITIONS)[0]
        distributions = self._distributions(definition)
        rows: list[SyntheticCoverageResult] = []
        for cell in definition.coupled_cells:
            alpha = float(cell.value(ExperimentAxisId.ALPHA))
            sample_count = int(cell.value(ExperimentAxisId.CALIBRATION_N))
            for distribution in distributions:
                rows.append(
                    iid_readiness_validation(
                        ExperimentCode.S2,
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
        definition = self.registry.get(ExperimentId.TEMPORAL_DEPENDENCE)
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
        definition = self.registry.get(ExperimentId.CALIBRATION_SHIFT)
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
        definition = self.registry.get(ExperimentId.CALIBRATION_CONTAMINATION)
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
        definition = self.registry.get(ExperimentId.MISMATCH_POWER)
        cells = tuple(
            exact_mismatch_power(sample_count, true_fpr)
            for sample_count in self._int_values(definition, ExperimentAxisId.MISMATCH_N)
            for true_fpr in self._float_values(definition, ExperimentAxisId.TRUE_FPR)
        )
        return self._write(output, definition, cells, 0)
