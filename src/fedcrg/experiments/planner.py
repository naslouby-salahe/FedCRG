"""Experiment-plan construction."""

from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import ExperimentId
from fedcrg.core.ids import Sha256
from fedcrg.experiments.models import ExperimentPlan
from fedcrg.experiments.registry import ExperimentRegistry


class ExperimentPlanner:
    def __init__(self, registry: ExperimentRegistry | None = None) -> None:
        self.registry = registry or ExperimentRegistry()

    def create(
        self,
        experiment_id: ExperimentId,
        config: ExperimentConfig,
        model_seed: int,
        calibration_seed: int,
    ) -> ExperimentPlan:
        if model_seed not in config.randomness.model_seeds:
            raise ValueError(f"Model seed {model_seed} is not configured")
        if calibration_seed not in config.dataset.calibration_seeds:
            raise ValueError(f"Calibration seed {calibration_seed} is not configured")
        return ExperimentPlan(
            definition=self.registry.get(experiment_id),
            config_hash=Sha256(config.config_hash),
            model_seed=model_seed,
            calibration_seed=calibration_seed,
        )
