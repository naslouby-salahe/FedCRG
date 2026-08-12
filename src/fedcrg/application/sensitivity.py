"""Real-score R2-R9 sensitivity execution over one immutable score cache."""

from __future__ import annotations

from pathlib import Path

from fedcrg.application.evaluate import EvaluatePolicies
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import DataRole, PolicyId
from fedcrg.experiments.real_sensitivity import contaminate_benign_scores, source_order_blocks
from fedcrg.protocol.mismatch import (
    bonferroni_fleet_sensitivity,
    holm_directional_fleet_sensitivity,
)
from fedcrg.protocol.readiness import familywise_readiness_assurance
from fedcrg.scoring.models import RoleScores, ScoreManifest
from fedcrg.scoring.views import (
    CalibrationScoreViews,
    ClientCalibrationScores,
    truncate_view,
)


class RunRealSensitivities:
    """Execute R2-R9 without retraining or creating another physical score cache."""

    def __init__(self, evaluator: EvaluatePolicies | None = None) -> None:
        self.evaluator = evaluator or EvaluatePolicies()

    @staticmethod
    def _replace_view_role(
        views: CalibrationScoreViews,
        role: DataRole,
        sample_count: int,
    ) -> CalibrationScoreViews:
        clients: dict = {}
        for client_id, client in views.clients.items():
            roles = dict(client.roles)
            roles[role] = truncate_view(client.get(role), sample_count)
            clients[client_id] = ClientCalibrationScores(
                client_id,
                client.calibration_seed,
                client.mode,
                roles,
            )
        return CalibrationScoreViews(views.calibration_seed, views.mode, clients)

    def readiness_sample_size(
        self,
        config: ExperimentConfig,
        scores: ScoreManifest,
        prepared_root: Path,
        calibration_seed: int,
        output: Path,
    ) -> Path:
        base_views = self.evaluator.calibration_views(
            config, scores, calibration_seed, prepared_root=prepared_root
        )
        cells = []
        for sample_count in (500, 1000, 1400, 1415, 1416, 1500, 2000):
            views = self._replace_view_role(
                base_views, DataRole.CALIBRATION, sample_count
            )
            cells.append(
                {
                    "calibration_n": sample_count,
                    "evaluation": self.evaluator.to_serializable(
                        self.evaluator.evaluate(
                            config,
                            scores,
                            calibration_views=views,
                        )
                    ),
                }
            )
        atomic_write_json(output, {"experiment": "R2", "cells": cells})
        return output

    def mismatch_sample_size(
        self,
        config: ExperimentConfig,
        scores: ScoreManifest,
        prepared_root: Path,
        calibration_seed: int,
        output: Path,
    ) -> Path:
        base_views = self.evaluator.calibration_views(
            config, scores, calibration_seed, prepared_root=prepared_root
        )
        cells = []
        for sample_count in (736, 1000, 1500, 2000, 3000):
            views = self._replace_view_role(base_views, DataRole.MISMATCH, sample_count)
            cells.append(
                {
                    "mismatch_n": sample_count,
                    "evaluation": self.evaluator.to_serializable(
                        self.evaluator.evaluate(
                            config,
                            scores,
                            calibration_views=views,
                        )
                    ),
                }
            )
        atomic_write_json(output, {"experiment": "R3", "cells": cells})
        return output

    def tolerance(
        self,
        config: ExperimentConfig,
        scores: ScoreManifest,
        prepared_root: Path,
        calibration_seed: int,
        output: Path,
    ) -> Path:
        views = self.evaluator.calibration_views(
            config, scores, calibration_seed, prepared_root=prepared_root
        )
        cells = []
        for rho in (0.25, 0.50, 1.00):
            protocol = config.protocol.model_copy(update={"rho": rho})
            variant = config.model_copy(update={"protocol": protocol})
            cells.append(
                {
                    "rho": rho,
                    "evaluation": self.evaluator.to_serializable(
                        self.evaluator.evaluate(
                            variant,
                            scores,
                            calibration_views=views,
                        )
                    ),
                }
            )
        atomic_write_json(output, {"experiment": "R4", "cells": cells})
        return output

    def target_fpr(
        self,
        config: ExperimentConfig,
        scores: ScoreManifest,
        prepared_root: Path,
        calibration_seed: int,
        output: Path,
    ) -> Path:
        views = self.evaluator.calibration_views(
            config, scores, calibration_seed, prepared_root=prepared_root
        )
        cells = []
        for alpha in (0.005, 0.01, 0.02, 0.05):
            protocol = config.protocol.model_copy(update={"alpha": alpha})
            variant = config.model_copy(update={"protocol": protocol})
            cells.append(
                {
                    "alpha": alpha,
                    "evaluation": self.evaluator.to_serializable(
                        self.evaluator.evaluate(
                            variant,
                            scores,
                            calibration_views=views,
                        )
                    ),
                }
            )
        atomic_write_json(output, {"experiment": "R5", "cells": cells})
        return output

    def assurance(
        self,
        config: ExperimentConfig,
        scores: ScoreManifest,
        prepared_root: Path,
        calibration_seed: int,
        output: Path,
    ) -> Path:
        views = self.evaluator.calibration_views(
            config, scores, calibration_seed, prepared_root=prepared_root
        )
        cells = []
        for assurance in (0.90, 0.95, 0.99):
            protocol = config.protocol.model_copy(
                update={"readiness_assurance": assurance}
            )
            variant = config.model_copy(update={"protocol": protocol})
            cells.append(
                {
                    "readiness_assurance": assurance,
                    "evaluation": self.evaluator.to_serializable(
                        self.evaluator.evaluate(
                            variant,
                            scores,
                            calibration_views=views,
                        )
                    ),
                }
            )
        atomic_write_json(output, {"experiment": "R6", "cells": cells})
        return output

    def multiplicity(
        self,
        config: ExperimentConfig,
        scores: ScoreManifest,
        prepared_root: Path,
        calibration_seed: int,
        output: Path,
    ) -> Path:
        views = self.evaluator.calibration_views(
            config, scores, calibration_seed, prepared_root=prepared_root
        )
        primary = self.evaluator.protocol_results(
            config, scores, calibration_views=views
        )
        counts = {
            client_id: (
                result.mismatch.exceedance_count,
                result.mismatch.sample_count,
            )
            for client_id, result in primary.items()
        }
        bonferroni = bonferroni_fleet_sensitivity(counts, config.protocol.band)
        holm = holm_directional_fleet_sensitivity(counts, config.protocol.band)
        payload = {
            "experiment": "R7",
            "client_count": len(primary),
            "familywise_readiness_assurance": familywise_readiness_assurance(
                len(primary)
            ),
            "bonferroni_mismatch": [
                {
                    "client_id": item.client_id.value,
                    "outcome": item.outcome.value,
                    "p_low": item.low_p_value,
                    "p_high": item.high_p_value,
                }
                for item in bonferroni
            ],
            "holm_directional": [
                {
                    "client_id": item.client_id.value,
                    "outcome": item.outcome.value,
                    "p_low": item.low_p_value,
                    "p_high": item.high_p_value,
                }
                for item in holm
            ],
        }
        atomic_write_json(output, payload)
        return output

    def source_order_test_blocks(
        self,
        config: ExperimentConfig,
        scores: ScoreManifest,
        prepared_root: Path,
        calibration_seed: int,
        output: Path,
    ) -> Path:
        bundle = self.evaluator.evaluate(
            config,
            scores,
            calibration_seed=calibration_seed,
            prepared_root=prepared_root,
        )
        threshold_by_cell = {
            (row.policy, row.client_id): row.threshold
            for row in bundle.clients
            if row.threshold is not None
        }
        rows = []
        for client_id, client in sorted(scores.clients.items()):
            benign = client.scores[DataRole.BENIGN_TEST].values
            for block_index, block in enumerate(source_order_blocks(benign, 5), start=1):
                for policy in config.policies:
                    threshold = threshold_by_cell.get((policy, client_id))
                    if threshold is None:
                        continue
                    rows.append(
                        {
                            "client_id": client_id.value,
                            "block": block_index,
                            "policy": policy.value,
                            "benign_n": len(block),
                            "fpr": float((block > threshold).mean()),
                        }
                    )
        atomic_write_json(output, {"experiment": "R8", "rows": rows})
        return output

    def real_contamination(
        self,
        config: ExperimentConfig,
        scores: ScoreManifest,
        prepared_root: Path,
        calibration_seed: int,
        output: Path,
    ) -> Path:
        base_views = self.evaluator.calibration_views(
            config, scores, calibration_seed, prepared_root=prepared_root
        )
        cells = []
        for fraction in (0.001, 0.005, 0.01, 0.02, 0.05):
            clients: dict = {}
            for client_id, view in base_views.clients.items():
                attack = scores.clients[client_id].scores[DataRole.ATTACK_DEV].values
                roles = dict(view.roles)
                for offset, role in enumerate((DataRole.MISMATCH, DataRole.CALIBRATION)):
                    original = view.get(role)
                    contaminated = contaminate_benign_scores(
                        original.values,
                        attack,
                        fraction,
                        seed=config.randomness.attack_split_seed + offset,
                    )
                    roles[role] = RoleScores(
                        role=role,
                        values=contaminated,
                        client_id=client_id,
                        row_ids=original.row_ids,
                    )
                clients[client_id] = ClientCalibrationScores(
                    client_id,
                    view.calibration_seed,
                    view.mode,
                    roles,
                )
            contaminated_views = CalibrationScoreViews(
                base_views.calibration_seed,
                base_views.mode,
                clients,
            )
            method_only = config.model_copy(update={"policies": (PolicyId.FEDCRG,)})
            cells.append(
                {
                    "fraction": fraction,
                    "evaluation": self.evaluator.to_serializable(
                        self.evaluator.evaluate(
                            method_only,
                            scores,
                            calibration_views=contaminated_views,
                        )
                    ),
                }
            )
        atomic_write_json(output, {"experiment": "R9", "cells": cells})
        return output
