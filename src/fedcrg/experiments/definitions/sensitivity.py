"""Execution of the pre-registered real-score sensitivities R2-R9 and R12.

Includes the R14 DIAD training-schema-only feature-contract derivation, which is a
sensitivity variant of the external DIAD experiment rather than a distinct pipeline
step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fedcrg.artifacts.manifests import EligibilityManifestStore
from fedcrg.artifacts.json_io import atomic_write_json
from fedcrg.configuration.experiment_config import ExperimentConfig
from fedcrg.configuration.method_config import ProtocolConfig
from fedcrg.configuration.validate import validate_experiment_config
from fedcrg.datasets.diad import DiadAdapter
from fedcrg.datasets.feature_sensitivity import (
    NumericSafeFeatureContract,
    derive_numeric_safe_features,
)
from fedcrg.domain.enums import (
    CalibrationAssignmentMode,
    DataRole,
    DatasetFeatureContractId,
    ExperimentAxisId,
    ExperimentId,
    MultiplicityProcedure,
    PolicyEvaluationStatus,
    PolicyId,
)
from fedcrg.domain.identifiers import CalibrationSeed, ClientId
from fedcrg.evaluation.evaluation_results import EvaluationBundle
from fedcrg.experiments.experiment_definition import ParameterSetting, get_experiment_definition
from fedcrg.decision.calibration_readiness import familywise_readiness_assurance
from fedcrg.decision.mismatch_detection import (
    FleetMismatchDecision,
    bonferroni_fleet_sensitivity,
    holm_directional_fleet_sensitivity,
)
from fedcrg.decision.results import ClientEvaluationResult
from fedcrg.experiments.policy_evaluation import EvaluatePolicies
from fedcrg.scoring.cache import ScoreCache
from fedcrg.scoring.calibration_scores import (
    CalibrationScoreViews,
    ClientCalibrationScores,
    RoleScores,
    truncate_view,
)

# --- Real-score sensitivity kernels -------------------------------------------------


def source_order_blocks(scores: np.ndarray, block_count: int = 5) -> tuple[np.ndarray, ...]:
    values = np.asarray(scores, dtype=np.float64)
    if block_count <= 0 or len(values) < block_count:
        raise ValueError("Source-order block analysis needs at least one row per block")
    return tuple(
        np.asarray(block, dtype=np.float64) for block in np.array_split(values, block_count)
    )


def contaminate_benign_scores(
    benign: np.ndarray,
    attack_dev: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    values = np.asarray(benign, dtype=np.float64).copy()
    attacks = np.asarray(attack_dev, dtype=np.float64)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Contamination fraction must be in [0,1]")
    count = int(round(fraction * len(values)))
    if count == 0:
        return values
    if count > len(attacks):
        raise ValueError("Attack-development cache is too small for requested contamination")
    rng = np.random.Generator(np.random.PCG64(seed))
    target = rng.choice(len(values), size=count, replace=False)
    source = rng.choice(len(attacks), size=count, replace=False)
    values[target] = attacks[source]
    return values


# --- R14 DIAD training-schema-only feature contract ---------------------------------


class BuildDiadFeatureSensitivityContract:
    """Derive R14 features strictly from frozen eligible clients' training rows."""

    def build(
        self,
        data_root: Path,
        eligibility_manifest: Path,
        output: Path,
        train_count: int = 2000,
    ) -> NumericSafeFeatureContract:
        eligible = EligibilityManifestStore().load(eligibility_manifest).eligible_clients
        if not eligible:
            raise ValueError("R14 requires a frozen non-empty DIAD eligibility list")
        adapter = DiadAdapter(data_root)
        discovered = set(adapter.discover_clients())
        missing = tuple(client for client in eligible if client not in discovered)
        if missing:
            raise ValueError(
                "Frozen eligible clients are absent from the acquired DIAD source: "
                + ", ".join(client.value for client in missing)
            )
        training_frames = {
            client: adapter.load_training_schema(client, train_count) for client in eligible
        }
        contract = derive_numeric_safe_features(training_frames)
        atomic_write_json(
            output,
            {
                "experiment": "R14",
                "source": "training_schema_only",
                "eligible_clients": [client.value for client in eligible],
                "feature_contract": contract,
            },
        )
        return contract


def r14_config(
    base_config: ExperimentConfig, contract: NumericSafeFeatureContract
) -> ExperimentConfig:
    """Create the R14-only derived config after the training-schema feature freeze.

    Rebuilds the full validated payload and revalidates through the model instead of
    an unchecked partial-field update, so the DIAD-training-schema cross-field
    invariants (feature_contract/feature_names/feature_count) are enforced.
    """
    payload = base_config.model_dump(mode="python")
    payload["id"] = ExperimentId.DIAD_FEATURE_SENSITIVITY
    dataset = dict(payload["dataset"])
    dataset["feature_contract"] = DatasetFeatureContractId.DIAD_TRAINING_NUMERIC_SAFE
    dataset["feature_count"] = contract.dimension
    dataset["feature_names"] = contract.features
    dataset["calibration_seeds"] = (base_config.dataset.primary_calibration_seed,)
    payload["dataset"] = dataset
    detector = dict(payload["detector"])
    detector["hidden_dims"] = contract.encoder_hidden_dims
    payload["detector"] = detector
    payload["policies"] = (
        PolicyId.GLOBAL_QUANTILE,
        PolicyId.LOCAL_QUANTILE,
        PolicyId.SHRINKAGE,
        PolicyId.FEDCRG,
    )
    return ExperimentConfig.model_validate(payload)


# --- Explicit, typed experiment-config variants (no policy or protocol registry) ----


class ExperimentVariantFactory:
    """Rebuild complete configs instead of bypassing validators with ``model_copy``."""

    def protocol_variant(
        self,
        config: ExperimentConfig,
        *,
        experiment_id: ExperimentId | None = None,
        alpha: float | None = None,
        rho: float | None = None,
        readiness_assurance: float | None = None,
        mismatch_confidence: float | None = None,
    ) -> ExperimentConfig:
        protocol = ProtocolConfig(
            id=config.protocol.id,
            version=config.protocol.version,
            alpha=config.protocol.alpha if alpha is None else alpha,
            rho=config.protocol.rho if rho is None else rho,
            readiness_assurance=(
                config.protocol.readiness_assurance
                if readiness_assurance is None
                else readiness_assurance
            ),
            mismatch_confidence=(
                config.protocol.mismatch_confidence
                if mismatch_confidence is None
                else mismatch_confidence
            ),
            strict_exceedance=config.protocol.strict_exceedance,
            reject_calibration_ties=config.protocol.reject_calibration_ties,
        )
        variant = ExperimentConfig(
            id=config.id if experiment_id is None else experiment_id,
            protocol=protocol,
            dataset=config.dataset,
            detector=config.detector,
            training=config.training,
            randomness=config.randomness,
            statistics=config.statistics,
            policies=config.policies,
            outputs_root=config.outputs_root,
        )
        validate_experiment_config(variant)
        return variant

    def policy_subset(
        self,
        config: ExperimentConfig,
        policies: tuple[PolicyId, ...],
        *,
        experiment_id: ExperimentId | None = None,
    ) -> ExperimentConfig:
        variant = ExperimentConfig(
            id=config.id if experiment_id is None else experiment_id,
            protocol=config.protocol,
            dataset=config.dataset,
            detector=config.detector,
            training=config.training,
            randomness=config.randomness,
            statistics=config.statistics,
            policies=policies,
            outputs_root=config.outputs_root,
        )
        validate_experiment_config(variant)
        return variant


# --- R2-R9 real-score sensitivity envelopes ------------------------------------------


@dataclass(frozen=True, slots=True)
class SensitivityCell:
    settings: tuple[ParameterSetting, ...]
    config_hash: str
    evaluation: EvaluationBundle


@dataclass(frozen=True, slots=True)
class SensitivityEnvelope:
    experiment_id: ExperimentId
    model_seed: int
    calibration_seed: CalibrationSeed
    cells: tuple[SensitivityCell, ...]


@dataclass(frozen=True, slots=True)
class MultiplicityCell:
    procedure: MultiplicityProcedure
    readiness_results: tuple[ClientEvaluationResult, ...] = ()
    mismatch_results: tuple[FleetMismatchDecision, ...] = ()


@dataclass(frozen=True, slots=True)
class MultiplicityEnvelope:
    experiment_id: ExperimentId
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
    calibration_seed: CalibrationSeed
    cells: tuple[SourceOrderBlockCell, ...]


class RunRealSensitivities:
    def __init__(
        self,
        evaluator: EvaluatePolicies | None = None,
        variants: ExperimentVariantFactory | None = None,
        score_cache: ScoreCache | None = None,
    ) -> None:
        self.evaluator = evaluator or EvaluatePolicies()
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
        definition = get_experiment_definition(experiment_id)
        base_views = self._base_views(
            config,
            score_root,
            prepared_root,
            calibration_seed,
        )
        cells: list[SensitivityCell] = []
        for raw_value in definition.axis(axis_id).values:
            if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
                raise TypeError(f"{definition.id.value}/{axis_id.value} must be numeric")
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
        definition = get_experiment_definition(ExperimentId.READINESS_SAMPLE_SIZE)
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
        definition = get_experiment_definition(ExperimentId.MISMATCH_SAMPLE_SIZE)
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
        definition = get_experiment_definition(ExperimentId.MULTIPLICITY_SENSITIVITY)
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
                    readiness_assurance=familywise_readiness_assurance(
                        len(views.client_ids), config.statistics.familywise_alpha
                    ),
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
                            familywise_alpha=config.statistics.familywise_alpha,
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
                            familywise_alpha=config.statistics.familywise_alpha,
                        ),
                    )
                )
        return self._write(
            output,
            MultiplicityEnvelope(
                ExperimentId.MULTIPLICITY_SENSITIVITY,
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
        definition = get_experiment_definition(ExperimentId.SOURCE_ORDER_TEST)
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
        definition = get_experiment_definition(ExperimentId.REAL_CONTAMINATION)
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
                int(self.score_cache.load_descriptor(score_root).identity.model_seed),
                seed,
                tuple(cells),
            ),
        )


class RunSourceOrderCalibration:
    """R12: reassign the frozen reservoir in source order, never retrain or rescore."""

    def run(
        self,
        config: ExperimentConfig,
        prepared_root: Path,
        score_root: Path,
        output: Path,
    ) -> Path:
        evaluator = EvaluatePolicies()
        descriptor = ScoreCache().load_descriptor(score_root)
        bundle = evaluator.evaluate_from_cache(
            config,
            score_root,
            calibration_seed=config.dataset.primary_calibration_seed,
            mode=CalibrationAssignmentMode.SOURCE_ORDER,
            prepared_root=prepared_root,
        )
        atomic_write_json(
            output,
            {
                "experiment": ExperimentId.SOURCE_ORDER_CALIBRATION.value,
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
