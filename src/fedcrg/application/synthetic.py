"""Application service for the locked S1-S6 synthetic programme."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.models import ExperimentConfig
from fedcrg.experiments.synthetic import (
    contamination_validation,
    exact_mismatch_power,
    iid_readiness_validation,
)


class RunSyntheticExperiments:
    def run_s1(self, config: ExperimentConfig, output: Path) -> Path:
        rows = []
        for distribution in ("normal", "lognormal", "gamma2", "normal_mixture"):
            for sample_count in (500, 1000, 1400, 1415, 1416, 1500, 2000, 3000):
                rows.append(asdict(iid_readiness_validation(
                    distribution,
                    sample_count,
                    repetitions=10000,
                    alpha=config.protocol.alpha,
                    rho=config.protocol.rho,
                    assurance=config.protocol.readiness_assurance,
                    seed=config.randomness.synthetic_seed + sample_count,
                )))
        atomic_write_json(output, {"experiment": "S1", "cells": rows})
        return output

    def run_s5(self, config: ExperimentConfig, output: Path) -> Path:
        rows = []
        for fraction in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05):
            for direction in ("high", "low"):
                rows.append(asdict(contamination_validation(
                    fraction,
                    direction,
                    repetitions=10000,
                    seed=config.randomness.synthetic_seed + int(fraction * 1_000_000),
                )))
        atomic_write_json(output, {"experiment": "S5", "cells": rows})
        return output

    def run_s6(self, output: Path) -> Path:
        rows = [
            asdict(exact_mismatch_power(sample_count, true_fpr))
            for sample_count in (736, 1000, 1500, 2000, 3000)
            for true_fpr in (0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03)
        ]
        atomic_write_json(output, {"experiment": "S6", "monte_carlo_trials": 0, "cells": rows})
        return output

    def run_s2(self, config: ExperimentConfig, output: Path) -> Path:
        sample_counts = {
            0.005: (2860, 2861, 5722),
            0.02: (693, 694, 1388),
            0.05: (269, 270, 540),
        }
        rows = []
        for alpha, counts in sample_counts.items():
            for distribution in ("normal", "lognormal", "gamma2", "normal_mixture"):
                for sample_count in counts:
                    rows.append(asdict(iid_readiness_validation(
                        distribution,
                        sample_count,
                        repetitions=10000,
                        alpha=alpha,
                        rho=0.5,
                        assurance=0.95,
                        seed=config.randomness.synthetic_seed + sample_count,
                    )))
        atomic_write_json(output, {"experiment": "S2", "cells": rows})
        return output

    def run_s3(self, config: ExperimentConfig, output: Path) -> Path:
        from fedcrg.analysis.robustness import temporal_dependence_stress

        rows = [
            asdict(temporal_dependence_stress(
                phi,
                sample_count,
                10000,
                config.protocol.band,
                config.protocol.readiness_assurance,
                config.randomness.synthetic_seed + sample_count + int(phi * 1000),
            ))
            for phi in (0.0, 0.3, 0.6, 0.9)
            for sample_count in (1416, 2000, 3000)
        ]
        atomic_write_json(output, {"experiment": "S3", "cells": rows})
        return output

    def run_s4(self, config: ExperimentConfig, output: Path) -> Path:
        from fedcrg.analysis.robustness import calibration_shift_stress

        rows = [
            asdict(calibration_shift_stress(
                shift,
                10000,
                config.protocol.band,
                config.protocol.readiness_assurance,
                config.randomness.synthetic_seed + int(shift * 1000),
            ))
            for shift in (0.0, 0.10, 0.25, 0.50, 1.00)
        ]
        atomic_write_json(output, {"experiment": "S4", "cells": rows})
        return output
