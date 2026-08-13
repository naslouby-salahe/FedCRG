"""Audited end-to-end execution of frozen detector and policy workloads from prepared data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fedcrg.configuration.experiment_config import ExperimentConfig
from fedcrg.domain.enums import ExperimentId
from fedcrg.domain.identifiers import CalibrationSeed, ModelSeed
from fedcrg.scoring.compute_scores import ComputeScores
from fedcrg.experiments.preflight import PreflightResult, ResearchPreflight
from fedcrg.experiments.policy_cells import FederationCellMaterializer, FrozenCacheInputs
from fedcrg.experiments.model_training import TrainDetector


@dataclass(frozen=True, slots=True)
class FrozenModelEvidence:
    model_seed: ModelSeed
    model_path: Path
    training_manifest: Path
    score_root: Path


@dataclass(frozen=True, slots=True)
class WorkloadExecution:
    experiment_id: ExperimentId
    models: tuple[FrozenModelEvidence, ...]
    run_directories: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ResearchExecution:
    preflight: PreflightResult
    workload: WorkloadExecution


class RunAllExperiments:
    """Audit prepared evidence, then train/score each model seed once and materialize
    every requested policy cell -- the single execution spine from prepared data to
    completed policy runs."""

    def __init__(
        self,
        preflight: ResearchPreflight | None = None,
        trainer: TrainDetector | None = None,
        scorer: ComputeScores | None = None,
        federation_cells: FederationCellMaterializer | None = None,
    ) -> None:
        self.preflight = preflight or ResearchPreflight()
        self.trainer = trainer or TrainDetector()
        self.scorer = scorer or ComputeScores()
        self.federation_cells = federation_cells or FederationCellMaterializer()

    def execute(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        prepared_root: Path,
        *,
        calibration_seeds: tuple[int, ...] | None = None,
    ) -> ResearchExecution:
        preflight = self.preflight.run(config, prepared_root)
        workload = self._execute_workload(
            experiment_id,
            config,
            prepared_root,
            calibration_seeds=calibration_seeds,
        )
        return ResearchExecution(preflight=preflight, workload=workload)

    def _execute_workload(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        prepared_root: Path,
        *,
        calibration_seeds: tuple[int, ...] | None = None,
    ) -> WorkloadExecution:
        """Execute the configured model/calibration grid without duplicate training or scoring."""

        seed_values = calibration_seeds or config.dataset.calibration_seeds
        if not seed_values:
            raise ValueError("At least one calibration seed is required")
        invalid = tuple(
            seed for seed in seed_values if seed not in config.dataset.calibration_seeds
        )
        if invalid:
            raise ValueError(
                f"Calibration seeds are outside the frozen dataset registry: {invalid}"
            )
        calibration_grid = tuple(CalibrationSeed(seed) for seed in seed_values)

        model_evidence: list[FrozenModelEvidence] = []
        run_directories: list[Path] = []
        for model_seed_value in config.randomness.model_seeds:
            model_seed = ModelSeed(model_seed_value)
            model_path, training_manifest = self.trainer.train_from_cache(
                config,
                prepared_root,
                model_seed,
            )
            score_root = self.scorer.score_from_cache(
                config,
                prepared_root,
                model_path,
                model_seed,
                training_manifest,
            )
            model_evidence.append(
                FrozenModelEvidence(
                    model_seed=model_seed,
                    model_path=model_path,
                    training_manifest=training_manifest,
                    score_root=score_root,
                )
            )
            caches = FrozenCacheInputs(
                prepared_root=prepared_root,
                model_path=model_path,
                training_manifest=training_manifest,
                score_root=score_root,
            )
            for calibration_seed in calibration_grid:
                cell = self.federation_cells.materialize(
                    experiment_id=experiment_id,
                    config=config,
                    model_seed=model_seed,
                    calibration_seed=calibration_seed,
                    caches=caches,
                )
                run_directories.extend(entry.path for entry in cell.run_directories)

        return WorkloadExecution(
            experiment_id=experiment_id,
            models=tuple(model_evidence),
            run_directories=tuple(run_directories),
        )
