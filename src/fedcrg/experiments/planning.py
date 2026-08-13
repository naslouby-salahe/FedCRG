"""Experiment-plan construction from one fully validated configuration."""

from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.domain.enums import ExperimentId
from fedcrg.domain.identifiers import CalibrationSeed, ModelSeed, Sha256
from fedcrg.experiments.execution import ExperimentPlan
from fedcrg.experiments.experiment_definition import get_experiment_definition


class ExperimentPlanner:
    def create(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: ModelSeed | int,
        calibration_seed: CalibrationSeed | int,
    ) -> ExperimentPlan:
        if config.id is not experiment_id:
            raise ValueError(
                f"Experiment identity mismatch: plan={experiment_id.value}, "
                f"config={config.id.value}"
            )
        typed_model_seed = ModelSeed(int(model_seed))
        typed_calibration_seed = CalibrationSeed(int(calibration_seed))
        if int(typed_model_seed) not in config.randomness.model_seeds:
            raise ValueError(f"Model seed {int(typed_model_seed)} is not configured")
        if int(typed_calibration_seed) not in config.dataset.calibration_seeds:
            raise ValueError(f"Calibration seed {int(typed_calibration_seed)} is not configured")
        definition = get_experiment_definition(experiment_id)
        if definition.policies and not set(config.policies).issubset(definition.policies):
            extra = set(config.policies) - set(definition.policies)
            raise ValueError(
                "Config contains policies outside the experiment catalogue: "
                + ", ".join(sorted(item.value for item in extra))
            )
        return ExperimentPlan(
            definition=definition,
            config_hash=Sha256(config.config_hash),
            model_seed=typed_model_seed,
            calibration_seed=typed_calibration_seed,
        )
