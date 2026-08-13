"""Materialize immutable policy evidence from one frozen federation evaluation."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from fedcrg.application.evaluate import EvaluatePolicies
from fedcrg.artifacts.dataset import PreparedDatasetManifestStore
from fedcrg.artifacts.hashing import sha256_file
from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.references import CacheReference, CacheReferenceStore
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.artifacts.training import TrainingManifestStore
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import CalibrationAssignmentMode, DatasetId, PolicyId
from fedcrg.core.ids import CalibrationSeed, RunId, Sha256
from fedcrg.metrics.results import EvaluationBundle, FederationMetrics
from fedcrg.scoring.cache import ScoreCache


@dataclass(frozen=True, slots=True)
class FrozenCacheInputs:
    prepared_root: Path
    model_path: Path
    training_manifest: Path
    score_root: Path


class PolicyCellMaterializer:
    """Create policy evidence while sharing one frozen federation evaluation."""

    def __init__(
        self,
        evaluator: EvaluatePolicies | None = None,
        score_cache: ScoreCache | None = None,
        references: CacheReferenceStore | None = None,
        training_manifests: TrainingManifestStore | None = None,
        dataset_manifests: PreparedDatasetManifestStore | None = None,
    ) -> None:
        self.evaluator = evaluator or EvaluatePolicies()
        self.score_cache = score_cache or ScoreCache()
        self.references = references or CacheReferenceStore()
        self.training_manifests = training_manifests or TrainingManifestStore()
        self.dataset_manifests = dataset_manifests or PreparedDatasetManifestStore()

    def evaluate_federation(
        self,
        config: ExperimentConfig,
        caches: FrozenCacheInputs,
        calibration_seed: CalibrationSeed | int,
        assignment_mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> EvaluationBundle:
        self.validate_upstream(config, caches)
        descriptor = self.score_cache.load_descriptor(caches.score_root)
        if descriptor.identity.data_spec_hash != Sha256(config.data_spec_hash):
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: data specification differs")
        if descriptor.identity.training_spec_hash != Sha256(config.training_spec_hash):
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: training specification differs")
        return self.evaluator.evaluate_from_cache(
            config,
            caches.score_root,
            calibration_seed=calibration_seed,
            mode=assignment_mode,
            prepared_root=caches.prepared_root,
        )

    def materialize(
        self,
        config: ExperimentConfig,
        policy: PolicyId,
        layout: RunLayout,
        caches: FrozenCacheInputs,
        calibration_seed: CalibrationSeed | int,
        assignment_mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> FederationMetrics | None:
        bundle = self.evaluate_federation(config, caches, calibration_seed, assignment_mode)
        return self.materialize_precomputed(
            config,
            policy,
            layout,
            caches,
            calibration_seed,
            bundle,
            assignment_mode,
        )

    def materialize_precomputed(
        self,
        config: ExperimentConfig,
        policy: PolicyId,
        layout: RunLayout,
        caches: FrozenCacheInputs,
        calibration_seed: CalibrationSeed | int,
        bundle: EvaluationBundle,
        assignment_mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> FederationMetrics | None:
        seed = CalibrationSeed(int(calibration_seed))
        self._copy_manifests(config, layout, caches, seed, assignment_mode)
        self._write_cache_references(config, layout, caches)
        descriptor = self.score_cache.load_descriptor(caches.score_root)
        self.evaluator.write_policy_artifacts(
            layout.root,
            RunId(layout.root.name),
            policy,
            bundle,
        )
        atomic_write_json(
            layout.reports / "evaluation_summary.json",
            {
                "calibration_seed": int(seed),
                "calibration_assignment": assignment_mode,
                "score_cache_sha256": descriptor.cache_sha256,
                "evaluation": self.evaluator.to_serializable(bundle),
            },
        )
        return next((item for item in bundle.federations if item.policy is policy), None)

    def validate_upstream(
        self,
        config: ExperimentConfig,
        caches: FrozenCacheInputs,
    ) -> None:
        prepared_manifest_path = caches.prepared_root / "manifest.json"
        preprocessing_path = caches.prepared_root / "preprocessing.json"
        required = (
            prepared_manifest_path,
            preprocessing_path,
            caches.model_path,
            caches.training_manifest,
            caches.score_root / ScoreCache.filename,
            caches.score_root / ScoreCache.manifest_filename,
        )
        missing = tuple(path for path in required if not path.is_file())
        if missing:
            raise FileNotFoundError(
                "Missing frozen upstream artifact(s): " + ", ".join(str(path) for path in missing)
            )

        prepared = self.dataset_manifests.load(prepared_manifest_path)
        if prepared.data_spec_hash != Sha256(config.data_spec_hash):
            raise ValueError("Prepared dataset provenance does not match requested cell")
        if prepared.dataset_id is not config.dataset.id:
            raise ValueError("Prepared dataset identity does not match requested cell")

        training = self.training_manifests.load(caches.training_manifest)
        if training.data_spec_hash != Sha256(config.data_spec_hash):
            raise ValueError("Training data specification does not match requested cell")
        if training.training_spec_hash != Sha256(config.training_spec_hash):
            raise ValueError("Training specification does not match requested cell")
        if training.dataset_manifest_sha256 != Sha256(sha256_file(prepared_manifest_path)):
            raise ValueError("Training manifest references a different dataset manifest")
        if training.preprocessing_sha256 != Sha256(sha256_file(preprocessing_path)):
            raise ValueError("Training manifest references a different preprocessing artifact")
        if training.model_file_sha256 != Sha256(sha256_file(caches.model_path)):
            raise ValueError("Frozen model hash does not match its training manifest")

        descriptor = self.score_cache.load_descriptor(caches.score_root)
        if descriptor.identity.data_spec_hash != training.data_spec_hash:
            raise ValueError("Score cache references a different data specification")
        if descriptor.identity.training_spec_hash != training.training_spec_hash:
            raise ValueError("Score cache references a different training specification")
        if descriptor.identity.dataset_manifest_hash != training.dataset_manifest_sha256:
            raise ValueError("Score cache references a different dataset manifest")
        if descriptor.identity.preprocessing_hash != training.preprocessing_sha256:
            raise ValueError("Score cache references a different preprocessing artifact")
        if descriptor.identity.model_hash != training.result.final_model_hash:
            raise ValueError("Score-cache model state does not match training result")

    @staticmethod
    def _copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(f".{destination.name}.tmp")
        shutil.copyfile(source, temp)
        temp.replace(destination)

    def _copy_manifests(
        self,
        config: ExperimentConfig,
        layout: RunLayout,
        caches: FrozenCacheInputs,
        calibration_seed: CalibrationSeed,
        assignment_mode: CalibrationAssignmentMode,
    ) -> None:
        self._copy(
            caches.prepared_root / "manifest.json",
            layout.data / "dataset_manifest.json",
        )
        self._copy(
            caches.prepared_root / "preprocessing.json",
            layout.data / "preprocessing.json",
        )
        eligibility_name = (
            "diad_eligibility.json" if config.dataset.id is DatasetId.DIAD else "eligibility.json"
        )
        eligibility_source = caches.prepared_root / eligibility_name
        if eligibility_source.exists():
            self._copy(eligibility_source, layout.data / "eligibility.json")
            if eligibility_name != "eligibility.json":
                self._copy(eligibility_source, layout.data / eligibility_name)
        assignment_source = (
            caches.prepared_root / "splits" / "seeded" / f"c{int(calibration_seed)}.json"
            if assignment_mode is CalibrationAssignmentMode.SEEDED_PERMUTATION
            else caches.prepared_root / "splits" / "source_order.json"
        )
        self._copy(
            assignment_source,
            layout.data / "calibration_assignment.json",
        )
        self._copy(
            caches.training_manifest,
            layout.training / "training.json",
        )
        self._copy(
            caches.score_root / ScoreCache.manifest_filename,
            layout.scores / "manifest.json",
        )

    def _write_cache_references(
        self,
        config: ExperimentConfig,
        layout: RunLayout,
        caches: FrozenCacheInputs,
    ) -> None:
        self.references.save(
            layout.model_reference,
            CacheReference.build(caches.model_path, config.outputs_root),
        )
        self.references.save(
            layout.score_reference,
            CacheReference.build(
                caches.score_root / ScoreCache.filename,
                config.outputs_root,
            ),
        )
