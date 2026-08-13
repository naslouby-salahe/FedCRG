"""Materialize all policy runs for one frozen federation/calibration cell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fedcrg.application.policy_cell import FrozenCacheInputs, PolicyCellMaterializer
from fedcrg.application.run_experiment import RunExperiment
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import CalibrationAssignmentMode, ExperimentId, PolicyId
from fedcrg.core.ids import CalibrationSeed, ModelSeed


@dataclass(frozen=True, slots=True)
class PolicyRunDirectory:
    policy: PolicyId
    path: Path


@dataclass(frozen=True, slots=True)
class FederationCellResult:
    experiment_id: ExperimentId
    model_seed: ModelSeed
    calibration_seed: CalibrationSeed
    run_directories: tuple[PolicyRunDirectory, ...]

    def directory_for(self, policy: PolicyId) -> Path:
        for entry in self.run_directories:
            if entry.policy is policy:
                return entry.path
        raise KeyError(policy.value)


class FederationCellMaterializer:
    """Evaluate the federation once, then write every requested policy cell."""

    def __init__(
        self,
        policy_cells: PolicyCellMaterializer | None = None,
        run_experiment: RunExperiment | None = None,
    ) -> None:
        self.policy_cells = policy_cells or PolicyCellMaterializer()
        self.run_experiment = run_experiment or RunExperiment()

    def materialize(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: ModelSeed | int,
        calibration_seed: CalibrationSeed | int,
        caches: FrozenCacheInputs,
        policies: tuple[PolicyId, ...] | None = None,
        assignment_mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> FederationCellResult:
        selected = config.policies if policies is None else policies
        if not selected:
            raise ValueError("At least one policy must be materialized")
        unknown = set(selected) - set(config.policies)
        if unknown:
            raise ValueError(
                "Requested policy is not configured: "
                + ", ".join(sorted(item.value for item in unknown))
            )

        typed_model_seed = ModelSeed(int(model_seed))
        typed_calibration_seed = CalibrationSeed(int(calibration_seed))
        bundle = self.policy_cells.evaluate_federation(
            config,
            caches,
            typed_calibration_seed,
            assignment_mode,
        )
        run_dirs: list[PolicyRunDirectory] = []
        for policy in selected:
            _, layout = self.run_experiment.execute(
                experiment_id=experiment_id,
                config=config,
                model_seed=int(typed_model_seed),
                calibration_seed=int(typed_calibration_seed),
                policy=policy,
                runner=lambda _plan, run_layout, p=policy: (
                    self.policy_cells.materialize_precomputed(
                        config,
                        p,
                        run_layout,
                        caches,
                        typed_calibration_seed,
                        bundle,
                        assignment_mode,
                    )
                ),
            )
            run_dirs.append(PolicyRunDirectory(policy, layout.root))
        return FederationCellResult(
            experiment_id=experiment_id,
            model_seed=typed_model_seed,
            calibration_seed=typed_calibration_seed,
            run_directories=tuple(run_dirs),
        )
