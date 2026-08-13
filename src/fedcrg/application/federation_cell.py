"""Materialize all policy runs for one frozen federation/calibration cell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fedcrg.application.policy_cell import FrozenCacheInputs, PolicyCellMaterializer
from fedcrg.application.run_experiment import RunExperiment
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import CalibrationAssignmentMode, ExperimentId, PolicyId


@dataclass(frozen=True, slots=True)
class FederationCellResult:
    experiment_id: ExperimentId
    model_seed: int
    calibration_seed: int
    run_directories: dict[PolicyId, Path]


class FederationCellMaterializer:
    """Evaluate the federation once, then write every requested policy cell."""

    def __init__(
        self,
        policy_cells: PolicyCellMaterializer | None = None,
    ) -> None:
        self.policy_cells = policy_cells or PolicyCellMaterializer()

    def materialize(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: int,
        calibration_seed: int,
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

        bundle = self.policy_cells.evaluate_federation(
            config,
            caches,
            calibration_seed,
            assignment_mode,
        )
        run_dirs: dict[PolicyId, Path] = {}
        for policy in selected:
            runner = RunExperiment()
            _, layout = runner.execute(
                experiment_id=experiment_id,
                config=config,
                model_seed=model_seed,
                calibration_seed=calibration_seed,
                policy=policy,
                runner=lambda _plan, run_layout, p=policy: (
                    self.policy_cells.materialize_precomputed(
                        config,
                        p,
                        run_layout,
                        caches,
                        calibration_seed,
                        bundle,
                        assignment_mode,
                    )
                ),
            )
            run_dirs[policy] = layout.root
        return FederationCellResult(
            experiment_id,
            model_seed,
            calibration_seed,
            run_dirs,
        )
