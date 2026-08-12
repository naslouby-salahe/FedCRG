"""Lifecycle orchestration for one immutable policy cell."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

import yaml

from fedcrg.artifacts.environment import capture_environment
from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.manifest import RunManifest, RunManifestStore
from fedcrg.artifacts.serialization import atomic_write_json, atomic_write_text
from fedcrg.artifacts.verification import ArtifactVerifier
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import ExperimentId, ExperimentStatus, PolicyId
from fedcrg.core.ids import RunId
from fedcrg.experiments.models import ExperimentPlan
from fedcrg.experiments.planner import ExperimentPlanner


class RunExperiment:
    def __init__(self) -> None:
        self.planner = ExperimentPlanner()
        self.manifests = RunManifestStore()
        self.verifier = ArtifactVerifier()

    @staticmethod
    def _manifest(layout: RunLayout, plan: ExperimentPlan, policy: PolicyId, status: ExperimentStatus) -> RunManifest:
        return RunManifest(
            RunId(layout.root.name),
            plan.definition.id,
            policy,
            plan.config_hash,
            plan.model_seed,
            plan.calibration_seed,
            status,
        )

    def prepare(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: int,
        calibration_seed: int,
        policy: PolicyId,
        repository_root: Path = Path("."),
    ) -> tuple[ExperimentPlan, RunLayout]:
        if policy not in config.policies:
            raise ValueError(f"Policy {policy.value} is not configured for this experiment")
        plan = self.planner.create(experiment_id, config, model_seed, calibration_seed)
        run_id = RunId.for_policy_cell(config, model_seed, calibration_seed, policy)
        layout = RunLayout.for_run(config.outputs_root, run_id)
        layout.create()
        self.manifests.save(layout.manifest, self._manifest(layout, plan, policy, ExperimentStatus.READY))
        atomic_write_text(layout.resolved_config, yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
        environment = capture_environment(repository_root)
        atomic_write_json(layout.environment, environment)
        atomic_write_json(layout.run_config, {
            "run_id": str(run_id),
            "experiment_id": experiment_id.value,
            "policy_id": policy.value,
            "parameters": config.model_dump(mode="json"),
            "model_seed": model_seed,
            "calibration_seed": calibration_seed,
            "config_hash": config.config_hash,
            "git_commit": environment["git_commit"],
            "environment_lock_sha256": environment["environment_lock_sha256"],
        })
        return plan, layout

    def execute(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: int,
        calibration_seed: int,
        policy: PolicyId,
        runner: Callable[[ExperimentPlan, RunLayout], object],
    ) -> tuple[object, RunLayout]:
        plan, layout = self.prepare(experiment_id, config, model_seed, calibration_seed, policy)
        self.manifests.save(layout.manifest, self._manifest(layout, plan, policy, ExperimentStatus.RUNNING))
        try:
            result = runner(plan, layout)
            self.manifests.save(layout.manifest, self._manifest(layout, plan, policy, ExperimentStatus.VERIFYING))
            verification = self.verifier.record(layout, plan.definition)
            if not verification.valid:
                raise RuntimeError(f"Run verification failed: {verification.missing}")
        except Exception:
            self.manifests.save(layout.manifest, self._manifest(layout, plan, policy, ExperimentStatus.FAILED))
            raise
        self.manifests.save(layout.manifest, self._manifest(layout, plan, policy, ExperimentStatus.COMPLETE))
        return result, layout
