"""R12 fixed source-order calibration-role sensitivity over the primary score cache."""

from __future__ import annotations

from pathlib import Path

from fedcrg.application.evaluate import EvaluatePolicies
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.domain.enums import CalibrationAssignmentMode, ExperimentCode
from fedcrg.scoring.cache import ScoreCache


class RunSourceOrderCalibration:
    """Reassign the frozen reservoir in source order, never retrain or rescore."""

    def run(
        self,
        config: ExperimentConfig,
        prepared_root: Path,
        score_root: Path,
        output: Path,
    ) -> Path:
        descriptor = ScoreCache().load_descriptor(score_root)
        bundle = EvaluatePolicies().evaluate_from_cache(
            config,
            score_root,
            calibration_seed=config.dataset.primary_calibration_seed,
            mode=CalibrationAssignmentMode.SOURCE_ORDER,
            prepared_root=prepared_root,
        )
        atomic_write_json(
            output,
            {
                "experiment": ExperimentCode.R12.value,
                "complete": True,
                "dataset_id": config.dataset.id.value,
                "model_seed": int(descriptor.identity.model_seed),
                "calibration_seed": config.dataset.primary_calibration_seed,
                "calibration_assignment": CalibrationAssignmentMode.SOURCE_ORDER.value,
                "score_cache_sha256": descriptor.cache_sha256.value,
                "evaluation": EvaluatePolicies.to_serializable(bundle),
            },
        )
        return output
