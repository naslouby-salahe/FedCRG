"""Execution of the pre-registered real-score sensitivities R2-R9 and R12."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fedcrg.application.evaluate import EvaluatePolicies
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.config.variants import ExperimentVariantFactory
from fedcrg.domain.enums import (
    CalibrationAssignmentMode,
    DataRole,
    ExperimentAxisId,
    ExperimentCode,
    ExperimentId,
    MultiplicityProcedure,
    PolicyEvaluationStatus,
    PolicyId,
)
from fedcrg.domain.identifiers import CalibrationSeed, ClientId
from fedcrg.experiments.models import ParameterSetting
from fedcrg.experiments.real_sensitivity import (
    contaminate_benign_scores,
    source_order_blocks,
)
from fedcrg.experiments.registry import ExperimentRegistry
from fedcrg.metrics.results import EvaluationBundle
from fedcrg.protocol.mismatch import (
    FleetMismatchDecision,
    bonferroni_fleet_sensitivity,
    holm_directional_fleet_sensitivity,
)
from fedcrg.protocol.readiness import familywise_readiness_assurance
from fedcrg.protocol.results import ClientProtocolResult
from fedcrg.scoring.cache import ScoreCache
from fedcrg.scoring.calibration_scores import (
    CalibrationScoreViews,
    ClientCalibrationScores,
    RoleScores,
    truncate_view,
)


@dataclass(frozen=True, slots=True)
class SensitivityCell:
    settings: tuple[ParameterSetting, ...]
    config_hash: str
    evaluation: EvaluationBundle


@dataclass(frozen=True, slots=True)
class SensitivityEnvelope:
    experiment_id: ExperimentId
    protocol_code: ExperimentCode
    model_seed: int
    calibration_seed: CalibrationSeed
    cells: tuple[SensitivityCell, ...]


@dataclass(frozen=True, slots=True)
class MultiplicityCell:
    procedure: MultiplicityProcedure
    readiness_results: tuple[ClientProtocolResult, ...] = ()
    mismatch_results: tuple[FleetMismatchDecision, ...] = ()


@dataclass(frozen=True, slots=True)
class MultiplicityEnvelope:
    experiment_id: ExperimentId
    protocol_code: ExperimentCode
    calibration_seed: CalibrationSeed
    cells: tuple[MultiplicityCell, ...]


@dataclass(frozen=True, slots=True)
class SourceOrderBlockCell:
    client_id: ClientId
    policy: PolicyId
    block_index: int
    block_count: int
    benign_n: int
    fpr: float | None


@dataclass(frozen=True, slots=True)
class SourceOrderEnvelope:
    experiment_id: ExperimentId
    protocol_code: ExperimentCode
    calibration_seed: CalibrationSeed
    cells: tuple[SourceOrderBlockCell, ...]


class RunRealSensitivities:
    def __init__(
        self,
        evaluator: EvaluatePolicies | None = None,
        registry: ExperimentRegistry | None = None,
        variants: ExperimentVariantFactory | None = None,
        score_cache: ScoreCache | None = None,
    ) -> None:
        self.evaluator = evaluator or EvaluatePolicies()
        self.registry = registry or ExperimentRegistry()
        self.variants = variants or ExperimentVariantFactory()
        self.score_cache = score_cache or ScoreCache()

    @staticmethod
    def _contaminate_role(
        item: RoleScores,
        attack_dev: np.ndarray,
        fraction: float,
        attack_split_seed: int,
        client_id: ClientId,
    ) -> RoleScores:
        if item.role not in {DataRole.MISMATCH, DataRole.CALIBRATION}:
            return item
        return RoleScores(
            role=item.role,
            values=contaminate_benign_scores(
                item.values,
                attack_dev,
                fraction,
                attack_split_seed + int(fraction * 1_000_000),
            ),
            client_id=client_id,
            row_ids=item.row_ids,
        )

    @staticmethod
    def _seed(
        config: ExperimentConfig,
        calibration_seed: CalibrationSeed | int | None,
    ) -> CalibrationSeed:
        return CalibrationSeed(
            config.dataset.primary_calibration_seed
            if calibration_seed is None
            else int(calibration_seed)
        )

    @staticmethod
    def _write(output: Path, payload: object) -> Path:
        atomic_write_json(output, payload)
        return output

    def _base_views(
        self,
        config: ExperimentConfig,
        score_root: Path,
        prepared_root: Path,
        calibration_seed: CalibrationSeed,
        mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> CalibrationScoreViews:
        return self.evaluator.calibration_views(
            config,
            score_root,
            calibration_seed,
            mode,
            prepared_root,
        )

    @staticmethod
    def _resize_views(
        views: CalibrationScoreViews,
        *,
        calibration_n: int | None = None,
        mismatch_n: int | None = None,
    ) -> CalibrationScoreViews:
        clients: list[ClientCalibrationScores] = []
        for client_id in views.client_ids:
            source = views.client(client_id)

            def _resize(item: RoleScores) -> RoleScores:
                if item.role is DataRole.CALIBRATION and calibration_n is not None:
                    return truncate_view(item, calibration_n)
                if item.role is DataRole.MISMATCH and mismatch_n is not None:
                    return truncate_view(item, mismatch_n)
                return item

            roles = tuple(_resize(item) for item in source.roles)
            clients.append(
                ClientCalibrationScores(
                    client_id,
                    source.calibration_seed,
                    source.mode,
                    roles,
                )
            )
        return CalibrationScoreViews(views.calibration_seed, views.mode, tuple(clients))

    def _parameter_sweep(
        self,
        *,
        experiment_id: ExperimentId,
        axis_id: ExperimentAxisId,
        config: ExperimentConfig,
        score_root: Path,
        prepared_root: Path,
        model_seed: int,
        calibration_seed: CalibrationSeed,
        output: Path,
    ) -> Path:
        definition = self.registry.get(experiment_id)
        base_views = self._base_views(
            config,
            score_root,
            prepared_root,
            calibration_seed,
        )
        cells: list[SensitivityCell] = []
        for raw_value in definition.axis(axis_id).values:
            if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
                raise TypeError(f"{definition.protocol_code.value}/{axis_id.value} must be numeric")
            value = float(raw_value)
            if axis_id is ExperimentAxisId.RHO:
                variant = self.variants.protocol_variant(
                    config,
                    experiment_id=experiment_id,
                    rho=value,
                )
            elif axis_id is ExperimentAxisId.ALPHA:
                variant = self.variants.protocol_variant(
                    config,
                    experiment_id=experiment_id,
                    alpha=value,
                )
            elif axis_id is ExperimentAxisId.READINESS_ASSURANCE:
                variant = self.variants.protocol_variant(
                    config,
                    experiment_id=experiment_id,
                    readiness_assurance=value,
                )
            else:
                raise ValueError(f"Unsupported protocol sensitivity axis: {axis_id.value}")
            evaluation = self.evaluator.evaluate_from_cache(
                variant,
                score_root,
                prepared_root=prepared_root,
                calibration_views=base_views,
            )
            cells.append(
                SensitivityCell(
                    settings=(ParameterSetting(axis_id, value),),
                    config_hash=variant.config_hash,
                    evaluation=evaluation,
                )
            )
        return self._write(
            output,
            SensitivityEnvelope(
                experiment_id,
                definition.protocol_code,
                model_seed,
                calibration_seed,
                tuple(cells),
            ),
        )

    def run_r2(
        self,
        config: ExperimentConfig,
        score_root: Path,
        prepared_root: Path,
        model_seed: int,
        output: Path,
        calibration_seed: CalibrationSeed | int | None = None,
    ) -> Path:
        seed = self._seed(config, calibration_seed)
        definition = self.registry.get(ExperimentId.READINESS_SAMPLE_SIZE)
        base_views = self._base_views(config, score_root, prepared_root, seed)
        variant = self.variants.protocol_variant(
            config,
            experiment_id=ExperimentId.READINESS_SAMPLE_SIZE,
        )
        cells = tuple(
            SensitivityCell(
                settings=(ParameterSetting(ExperimentAxisId.CALIBRATION_N, int(value)),),
                config_hash=variant.config_hash,
                evaluation=self.evaluator.evaluate_from_cache(
                    variant,
                    score_root,
                    prepared_root=prepared_root,
                    calibration_views=self._resize_views(
                        base_views,
                        calibration_n=int(value),
                    ),
                ),
            )
            for value in definition.axis(ExperimentAxisId.CALIBRATION_N).values
        )
        return self._write(
            output,
            SensitivityEnvelope(
                ExperimentId.READINESS_SAMPLE_SIZE,
                definition.protocol_code,
                model_seed,
                seed,
                cells,
            ),
        )

    def run_r3(
        self,
        config: ExperimentConfig,
        score_root: Path,
        prepared_root: Path,
        model_seed: int,
        output: Path,
        calibration_seed: CalibrationSeed | int | None = None,
    ) -> Path:
        seed = self._seed(config, calibration_seed)
        definition = self.registry.get(ExperimentId.MISMATCH_SAMPLE_SIZE)
        base_views = self._base_views(config, score_root, prepared_root, seed)
        variant = self.variants.protocol_variant(
            config,
            experiment_id=ExperimentId.MISMATCH_SAMPLE_SIZE,
        )
        cells = tuple(
            SensitivityCell(
                settings=(ParameterSetting(ExperimentAxisId.MISMATCH_N, int(value)),),
                config_hash=variant.config_hash,
                evaluation=self.evaluator.evaluate_from_cache(
                    variant,
                    score_root,
                    prepared_root=prepared_root,
                    calibration_views=self._resize_views(
                        base_views,
                        mismatch_n=int(value),
                    ),
                ),
            )
            for value in definition.axis(ExperimentAxisId.MISMATCH_N).values
        )
        return self._write(
            output,
            SensitivityEnvelope(
                ExperimentId.MISMATCH_SAMPLE_SIZE,
                definition.protocol_code,
                model_seed,
                seed,
                cells,
            ),
        )

    def run_r4(
        self,
        config: ExperimentConfig,
        score_root: Path,
        prepared_root: Path,
        model_seed: int,
        output: Path,
        calibration_seed: CalibrationSeed | int | None = None,
    ) -> Path:
        seed = self._seed(config, calibration_seed)
        return self._parameter_sweep(
            experiment_id=ExperimentId.TOLERANCE_SENSITIVITY,
            axis_id=ExperimentAxisId.RHO,
            config=config,
            score_root=score_root,
            prepared_root=prepared_root,
            model_seed=model_seed,
            calibration_seed=seed,
            output=output,
        )

    def run_r5(
        self,
        config: ExperimentConfig,
        score_root: Path,
        prepared_root: Path,
        model_seed: int,
        output: Path,
        calibration_seed: CalibrationSeed | int | None = None,
    ) -> Path:
        seed = self._seed(config, calibration_seed)
        return self._parameter_sweep(
            experiment_id=ExperimentId.TARGET_FPR_REAL,
            axis_id=ExperimentAxisId.ALPHA,
            config=config,
            score_root=score_root,
            prepared_root=prepared_root,
            model_seed=model_seed,
            calibration_seed=seed,
            output=output,
        )

    def run_r6(
        self,
        config: ExperimentConfig,
        score_root: Path,
        prepared_root: Path,
        model_seed: int,
        output: Path,
        calibration_seed: CalibrationSeed | int | None = None,
    ) -> Path:
        seed = self._seed(config, calibration_seed)
        return self._parameter_sweep(
            experiment_id=ExperimentId.ASSURANCE_SENSITIVITY,
            axis_id=ExperimentAxisId.READINESS_ASSURANCE,
            config=config,
            score_root=score_root,
            prepared_root=prepared_root,
            model_seed=model_seed,
            calibration_seed=seed,
            output=output,
        )

    def run_r7(
        self,
        config: ExperimentConfig,
        score_root: Path,
        prepared_root: Path,
        output: Path,
        calibration_seed: CalibrationSeed | int | None = None,
    ) -> Path:
        seed = self._seed(config, calibration_seed)
        definition = self.registry.get(ExperimentId.MULTIPLICITY_SENSITIVITY)
        views = self._base_views(config, score_root, prepared_root, seed)
        fedcrg_only = self.variants.policy_subset(
            config,
            (PolicyId.FEDCRG,),
            experiment_id=ExperimentId.MULTIPLICITY_SENSITIVITY,
        )
        base_results = self.evaluator.protocol_results(fedcrg_only, views)
        counts = {
            client_id: (
                result.mismatch.exceedance_count,
                result.mismatch.sample_count,
            )
            for client_id, result in base_results.items()
        }
        cells: list[MultiplicityCell] = []
        for raw_procedure in definition.axis(ExperimentAxisId.PROCEDURE).values:
            if not isinstance(raw_procedure, MultiplicityProcedure):
                raise TypeError("R7 procedure axis is malformed")
            if raw_procedure is MultiplicityProcedure.BONFERRONI_READINESS:
                adjusted = self.variants.protocol_variant(
                    fedcrg_only,
                    experiment_id=ExperimentId.MULTIPLICITY_SENSITIVITY,
                    readiness_assurance=familywise_readiness_assurance(len(views.client_ids)),
                )
                readiness = tuple(self.evaluator.protocol_results(adjusted, views).values())
                cells.append(MultiplicityCell(raw_procedure, readiness_results=readiness))
            elif raw_procedure is MultiplicityProcedure.BONFERRONI_MISMATCH:
                cells.append(
                    MultiplicityCell(
                        raw_procedure,
                        mismatch_results=bonferroni_fleet_sensitivity(
                            counts,
                            config.protocol.band,
                        ),
                    )
                )
            else:
                cells.append(
                    MultiplicityCell(
                        raw_procedure,
                        mismatch_results=holm_directional_fleet_sensitivity(
                            counts,
                            config.protocol.band,
                        ),
                    )
                )
        return self._write(
            output,
            MultiplicityEnvelope(
                ExperimentId.MULTIPLICITY_SENSITIVITY,
                definition.protocol_code,
                seed,
                tuple(cells),
            ),
        )

    def run_r8(
        self,
        config: ExperimentConfig,
        score_root: Path,
        prepared_root: Path,
        output: Path,
        calibration_seed: CalibrationSeed | int | None = None,
    ) -> Path:
        seed = self._seed(config, calibration_seed)
        definition = self.registry.get(ExperimentId.SOURCE_ORDER_TEST)
        block_count = int(definition.axis(ExperimentAxisId.BLOCKS).values[0])
        variant = self.variants.protocol_variant(
            config,
            experiment_id=ExperimentId.SOURCE_ORDER_TEST,
        )
        bundle = self.evaluator.evaluate_from_cache(
            variant,
            score_root,
            calibration_seed=seed,
            prepared_root=prepared_root,
        )
        thresholds = {
            (row.client_id, row.policy): row.threshold
            for row in bundle.clients
            if row.status is PolicyEvaluationStatus.EVALUATED
        }
        rows: list[SourceOrderBlockCell] = []
        descriptor = self.score_cache.load_descriptor(score_root)
        for client_id in descriptor.client_ids:
            benign = self.score_cache.read_role(
                score_root,
                client_id,
                DataRole.BENIGN_TEST,
            ).values
            blocks = source_order_blocks(benign, block_count)
            for policy in variant.policies:
                threshold = thresholds.get((client_id, policy))
                for block_index, block in enumerate(blocks, start=1):
                    rows.append(
                        SourceOrderBlockCell(
                            client_id=client_id,
                            policy=policy,
                            block_index=block_index,
                            block_count=block_count,
                            benign_n=len(block),
                            fpr=(None if threshold is None else float(np.mean(block > threshold))),
                        )
                    )
        return self._write(
            output,
            SourceOrderEnvelope(
                ExperimentId.SOURCE_ORDER_TEST,
                definition.protocol_code,
                seed,
                tuple(rows),
            ),
        )

    def run_r9(
        self,
        config: ExperimentConfig,
        score_root: Path,
        prepared_root: Path,
        output: Path,
        calibration_seed: CalibrationSeed | int | None = None,
    ) -> Path:
        seed = self._seed(config, calibration_seed)
        definition = self.registry.get(ExperimentId.REAL_CONTAMINATION)
        variant = self.variants.policy_subset(
            config,
            (PolicyId.FEDCRG,),
            experiment_id=ExperimentId.REAL_CONTAMINATION,
        )
        base_views = self._base_views(config, score_root, prepared_root, seed)
        cells: list[SensitivityCell] = []
        for raw_fraction in definition.axis(ExperimentAxisId.FRACTION).values:
            fraction = float(raw_fraction)
            clients: list[ClientCalibrationScores] = []
            for client_id in base_views.client_ids:
                source = base_views.client(client_id)
                attack_dev = self.score_cache.read_role(
                    score_root,
                    client_id,
                    DataRole.ATTACK_DEV,
                ).values
                roles = tuple(
                    self._contaminate_role(
                        item, attack_dev, fraction, config.randomness.attack_split_seed, client_id
                    )
                    for item in source.roles
                )
                clients.append(
                    ClientCalibrationScores(
                        client_id,
                        seed,
                        base_views.mode,
                        roles,
                    )
                )
            contaminated = CalibrationScoreViews(seed, base_views.mode, tuple(clients))
            cells.append(
                SensitivityCell(
                    settings=(ParameterSetting(ExperimentAxisId.FRACTION, fraction),),
                    config_hash=variant.config_hash,
                    evaluation=self.evaluator.evaluate_from_cache(
                        variant,
                        score_root,
                        prepared_root=prepared_root,
                        calibration_views=contaminated,
                    ),
                )
            )
        return self._write(
            output,
            SensitivityEnvelope(
                ExperimentId.REAL_CONTAMINATION,
                definition.protocol_code,
                int(self.score_cache.load_descriptor(score_root).identity.model_seed),
                seed,
                tuple(cells),
            ),
        )

    def run_r12(
        self,
        config: ExperimentConfig,
        score_root: Path,
        prepared_root: Path,
        output: Path,
    ) -> Path:
        seed = CalibrationSeed(config.dataset.primary_calibration_seed)
        definition = self.registry.get(ExperimentId.SOURCE_ORDER_CALIBRATION)
        variant = self.variants.protocol_variant(
            config,
            experiment_id=ExperimentId.SOURCE_ORDER_CALIBRATION,
        )
        bundle = self.evaluator.evaluate_from_cache(
            variant,
            score_root,
            calibration_seed=seed,
            mode=CalibrationAssignmentMode.SOURCE_ORDER,
            prepared_root=prepared_root,
        )
        return self._write(
            output,
            SensitivityEnvelope(
                ExperimentId.SOURCE_ORDER_CALIBRATION,
                definition.protocol_code,
                int(self.score_cache.load_descriptor(score_root).identity.model_seed),
                seed,
                (
                    SensitivityCell(
                        settings=(
                            ParameterSetting(
                                ExperimentAxisId.ASSIGNMENT,
                                CalibrationAssignmentMode.SOURCE_ORDER,
                            ),
                        ),
                        config_hash=variant.config_hash,
                        evaluation=bundle,
                    ),
                ),
            ),
        )
