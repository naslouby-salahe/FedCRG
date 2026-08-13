"""R13 benchmark application using precomputed and cached protocol primitives."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from fedcrg.analysis.benchmark import benchmark, configure_single_thread_execution
from fedcrg.artifacts.environment import capture_environment
from fedcrg.artifacts.experiment_results import ExperimentResultEnvelope
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.ids import ClientId, Sha256
from fedcrg.protocol.decision import ThresholdDecisionEngine
from fedcrg.protocol.mismatch import ReferenceMismatchEvaluator
from fedcrg.protocol.readiness import CalibrationReadinessEvaluator, ReadinessPlanCache
from fedcrg.protocol.reference import ReferenceThresholdEstimator


class RunBenchmark:
    """Execute the four locked R13 primitive measurements."""

    def run_on_synthetic_evidence(
        self,
        config: ExperimentConfig,
        output: Path,
        repository_root: Path = Path("."),
        synthetic_seed: int = 123456,
    ) -> Path:
        """Build representative-sized synthetic benign evidence, then benchmark the primitives.

        R13 measures per-primitive wall time/memory at deployment-representative input
        sizes, it is not a claim about the frozen protocol's statistical behavior, so
        synthetic benign scores are sufficient input evidence.
        """

        rng = np.random.default_rng(synthetic_seed)
        client_count = max(config.dataset.expected_clients or 9, 1)
        references = {
            ClientId(f"benchmark-client-{i:03d}"): rng.normal(
                size=config.dataset.split.reference_benign
            ).astype(np.float64)
            for i in range(client_count)
        }
        calibration = rng.normal(size=config.dataset.split.calibration_benign).astype(np.float64)
        mismatch = rng.normal(size=config.dataset.split.mismatch_benign).astype(np.float64)

        reference_estimator = ReferenceThresholdEstimator()
        reference = reference_estimator.estimate(references, config.protocol.alpha)
        plan = ReadinessPlanCache().precompute(
            len(calibration), config.protocol.band, config.protocol.readiness_assurance
        )
        readiness_evaluator = CalibrationReadinessEvaluator()
        mismatch_evaluator = ReferenceMismatchEvaluator()
        decision_engine = ThresholdDecisionEngine()

        readiness = readiness_evaluator.evaluate(calibration, plan)
        mismatch_result = mismatch_evaluator.evaluate(
            mismatch, reference.value, config.protocol.band, config.protocol.mismatch_confidence
        )

        operations: dict[str, Callable[[], object]] = {
            "reference_construction": lambda: reference_estimator.estimate(
                references, config.protocol.alpha
            ),
            "readiness_lookup_and_order_statistic": lambda: readiness_evaluator.evaluate(
                calibration, plan
            ),
            "reference_mismatch_interval": lambda: mismatch_evaluator.evaluate(
                mismatch, reference.value, config.protocol.band, config.protocol.mismatch_confidence
            ),
            "policy_decision": lambda: decision_engine.decide(reference, readiness, mismatch_result),
        }
        return self.run(config, operations, output, repository_root)

    def run(
        self,
        config: ExperimentConfig,
        operations: dict[str, Callable[[], object]],
        output: Path,
        repository_root: Path = Path("."),
    ) -> Path:
        required = {
            "reference_construction",
            "readiness_lookup_and_order_statistic",
            "reference_mismatch_interval",
            "policy_decision",
        }
        if set(operations) != required:
            raise ValueError(
                "R13 requires exactly the four locked primitive operations: "
                + ", ".join(sorted(required))
            )
        pinned_cpu = configure_single_thread_execution()
        rows = []
        for name in sorted(operations):
            row = asdict(
                benchmark(
                    name,
                    operations[name],
                    warmups=100,
                    repetitions=1000,
                )
            )
            row["pinned_cpu"] = pinned_cpu
            rows.append(row)

        environment = capture_environment(repository_root)
        notes = (
            "Runtime values are empirical machine-specific evidence, not mathematical constants.",
        )
        return ExperimentResultEnvelope(
            protocol_code="R13",
            config_hash=Sha256(config.config_hash),
            master_seed=config.randomness.synthetic_seed,
            expected_cells=4,
            expected_monte_carlo_trials=0,
            expected_exact_cells=0,
            cells=tuple(rows),
            notes=notes,
            metadata={"environment": environment, "pinned_cpu": pinned_cpu},
        ).write(output)
