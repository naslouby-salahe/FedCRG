"""R12 fixed source-order calibration-role sensitivity over the primary score cache."""

from __future__ import annotations

from pathlib import Path

from fedcrg.application.evaluate import EvaluatePolicies
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import CalibrationAssignmentMode
from fedcrg.scoring.cache import ScoreCache


class RunSourceOrderCalibration:
    """Reassign the frozen reservoir in source order; never retrain or rescore."""

    def run(
        self,
        config: ExperimentConfig,
        prepared_root: Path,
        score_root: Path,
        output: Path,
    ) -> Path:
        scores = ScoreCache().load(score_root)
        bundle = EvaluatePolicies().evaluate(
            config,
            scores,
            calibration_seed=config.dataset.primary_calibration_seed,
            mode=CalibrationAssignmentMode.SOURCE_ORDER,
            prepared_root=prepared_root,
        )
        atomic_write_json(
            output,
            {
                "experiment": "R12",
                "dataset": config.dataset.id.value,
                "calibration_assignment": CalibrationAssignmentMode.SOURCE_ORDER.value,
                "score_cache_sha256": scores.cache_sha256.value if scores.cache_sha256 else None,
                "evaluation": EvaluatePolicies.to_serializable(bundle),
            },
        )
        return output
