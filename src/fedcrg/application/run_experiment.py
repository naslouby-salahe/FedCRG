"""Application-level experiment orchestration."""

from __future__ import annotations

import platform
import sys

import yaml

from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.manifest import RunManifest, RunManifestStore
from fedcrg.artifacts.serialization import atomic_write_json, atomic_write_text
from fedcrg.artifacts.verification import ArtifactVerifier
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import ExperimentId, ExperimentStatus
from fedcrg.core.ids import RunId
from fedcrg.experiments.models import ExperimentPlan, ExperimentRunner
from fedcrg.experiments.planner import ExperimentPlanner


class RunExperiment:
    def __init__(
        self,
        planner: ExperimentPlanner | None = None,
        manifest_store: RunManifestStore | None = None,
        verifier: ArtifactVerifier | None = None,
    ) -> None:
        self.planner = planner or ExperimentPlanner()
        self.manifest_store = manifest_store or RunManifestStore()
        self.verifier = verifier or ArtifactVerifier()

    def _manifest(
        self,
        layout: RunLayout,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: int,
        calibration_seed: int,
        status: ExperimentStatus,
    ) -> RunManifest:
        return RunManifest(
            run_id=layout.root.name,
            experiment_id=experiment_id.value,
            config_hash=config.config_hash,
            model_seed=model_seed,
            calibration_seed=calibration_seed,
            status=status,
        )

    def prepare(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: int,
        calibration_seed: int,
    ) -> tuple[ExperimentPlan, RunLayout]:
        plan = self.planner.create(experiment_id, config, model_seed, calibration_seed)
        run_id = RunId.derive(experiment_id, config.config_hash, model_seed, calibration_seed)
        layout = RunLayout.for_run(config.outputs_root, run_id)
        layout.create()
        self.manifest_store.save(
            layout.manifest,
            self._manifest(
                layout,
                experiment_id,
                config,
                model_seed,
                calibration_seed,
                ExperimentStatus.READY,
            ),
        )
        atomic_write_text(
            layout.resolved_config,
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        )
        atomic_write_json(
            layout.environment,
            {"python": sys.version, "platform": platform.platform()},
        )
        return plan, layout

    def execute(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: int,
        calibration_seed: int,
        runner: ExperimentRunner,
    ) -> tuple[object, RunLayout]:
        plan, layout = self.prepare(experiment_id, config, model_seed, calibration_seed)
        self.manifest_store.save(
            layout.manifest,
            self._manifest(
                layout,
                experiment_id,
                config,
                model_seed,
                calibration_seed,
                ExperimentStatus.RUNNING,
            ),
        )
        try:
            result = runner(plan)
            self.manifest_store.save(
                layout.manifest,
                self._manifest(
                    layout,
                    experiment_id,
                    config,
                    model_seed,
                    calibration_seed,
                    ExperimentStatus.VERIFYING,
                ),
            )
            verification = self.verifier.record(layout)
            if not verification.valid:
                raise RuntimeError(f"Run verification failed: {verification.missing}")
        except Exception:
            self.manifest_store.save(
                layout.manifest,
                self._manifest(
                    layout,
                    experiment_id,
                    config,
                    model_seed,
                    calibration_seed,
                    ExperimentStatus.FAILED,
                ),
            )
            raise
        self.manifest_store.save(
            layout.manifest,
            self._manifest(
                layout,
                experiment_id,
                config,
                model_seed,
                calibration_seed,
                ExperimentStatus.COMPLETE,
            ),
        )
        return result, layout
