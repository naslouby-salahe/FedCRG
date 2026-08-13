"""Lifecycle orchestration for one immutable policy cell."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml

from fedcrg.artifacts.environment import capture_environment
from fedcrg.artifacts.identity import RunIdentityFactory
from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.manifest import RunManifest, RunManifestStore
from fedcrg.artifacts.serialization import atomic_write_json, atomic_write_text
from fedcrg.artifacts.verification import ArtifactVerifier
from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.domain.enums import ExperimentId, ExperimentStatus, PolicyId
from fedcrg.domain.identifiers import RunId
from fedcrg.experiments.lifecycle import assert_transition
from fedcrg.experiments.models import ExperimentPlan
from fedcrg.experiments.planner import ExperimentPlanner


class RunExperiment:
    """Manage one policy-cell lifecycle without owning scientific computation."""

    def __init__(self) -> None:
        self.planner = ExperimentPlanner()
        self.manifests = RunManifestStore()
        self.verifier = ArtifactVerifier()

    @staticmethod
    def _manifest(
        layout: RunLayout,
        plan: ExperimentPlan,
        policy: PolicyId,
        status: ExperimentStatus,
    ) -> RunManifest:
        return RunManifest(
            run_id=RunId(layout.root.name),
            experiment_id=plan.definition.id,
            policy_id=policy,
            config_hash=plan.config_hash,
            model_seed=plan.model_seed,
            calibration_seed=plan.calibration_seed,
            status=status,
        )

    def _transition(
        self,
        layout: RunLayout,
        plan: ExperimentPlan,
        policy: PolicyId,
        target: ExperimentStatus,
    ) -> None:
        current = self.manifests.load(layout.manifest).status
        assert_transition(current, target)
        self.manifests.save(layout.manifest, self._manifest(layout, plan, policy, target))

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
        run_id = RunIdentityFactory.for_policy_cell(config, model_seed, calibration_seed, policy)
        layout = RunLayout.for_run(config.outputs_root, run_id)
        layout.create()
        self.manifests.save(
            layout.manifest,
            self._manifest(layout, plan, policy, ExperimentStatus.PENDING),
        )
        self._transition(layout, plan, policy, ExperimentStatus.VALIDATING)
        self._transition(layout, plan, policy, ExperimentStatus.READY)
        atomic_write_text(
            layout.resolved_config,
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        )
        environment = capture_environment(repository_root)
        atomic_write_json(layout.environment, environment)
        atomic_write_json(
            layout.run_config,
            {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "policy_id": policy,
                "parameters": config.model_dump(mode="json"),
                "model_seed": model_seed,
                "calibration_seed": calibration_seed,
                "config_hash": config.config_hash,
                "data_spec_hash": config.data_spec_hash,
                "training_spec_hash": config.training_spec_hash,
                "git_commit": environment["git_commit"],
                "git_clean": environment["git_clean"],
                "git_patch_sha256": environment["git_patch_sha256"],
                "environment_pin_sha256": environment["environment_pin_sha256"],
                "environment_pin_kind": environment["environment_pin_kind"],
            },
        )
        return plan, layout

    def execute(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: int,
        calibration_seed: int,
        policy: PolicyId,
        runner: Callable[[ExperimentPlan, RunLayout], object],
        repository_root: Path = Path("."),
    ) -> tuple[object, RunLayout]:
        plan, layout = self.prepare(
            experiment_id,
            config,
            model_seed,
            calibration_seed,
            policy,
            repository_root,
        )
        self._transition(layout, plan, policy, ExperimentStatus.RUNNING)
        try:
            result = runner(plan, layout)
            self._transition(layout, plan, policy, ExperimentStatus.VERIFYING)
            verification = self.verifier.record(layout, plan.definition)
            if not verification.valid:
                raise RuntimeError(
                    "Run verification failed: "
                    + ", ".join(verification.missing + verification.mismatched)
                )
        except Exception:
            self._transition(layout, plan, policy, ExperimentStatus.FAILED)
            raise
        self._transition(layout, plan, policy, ExperimentStatus.COMPLETE)
        return result, layout
