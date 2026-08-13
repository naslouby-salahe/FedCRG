"""R13 benchmark application using precomputed and cached protocol primitives."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fedcrg.analysis.benchmark import benchmark, configure_single_thread_execution
from fedcrg.artifacts.environment import capture_environment
from fedcrg.artifacts.experiment_results import ExperimentResultEnvelope
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.ids import Sha256


class RunBenchmark:
    """Execute the four locked R13 primitive measurements."""

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
