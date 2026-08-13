"""High-level audited execution path for real-data FedCRG workloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fedcrg.application.pipeline import ExecuteFrozenWorkload, WorkloadExecution
from fedcrg.application.preflight import PreflightResult, ResearchPreflight
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import ExperimentId


@dataclass(frozen=True, slots=True)
class ResearchExecution:
    preflight: PreflightResult
    workload: WorkloadExecution


class ExecuteResearchPipeline:
    """Run only from audited prepared evidence and frozen statistical tables."""

    def __init__(
        self,
        preflight: ResearchPreflight | None = None,
        workload: ExecuteFrozenWorkload | None = None,
    ) -> None:
        self.preflight = preflight or ResearchPreflight()
        self.workload = workload or ExecuteFrozenWorkload()

    def execute(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        prepared_root: Path,
        *,
        calibration_seeds: tuple[int, ...] | None = None,
    ) -> ResearchExecution:
        preflight = self.preflight.run(config, prepared_root)
        workload = self.workload.execute(
            experiment_id,
            config,
            prepared_root,
            calibration_seeds=calibration_seeds,
        )
        return ResearchExecution(preflight=preflight, workload=workload)
