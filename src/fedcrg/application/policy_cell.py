"""Materialize one immutable policy cell from frozen dataset/model/score caches."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from fedcrg.application.evaluate import EvaluatePolicies
from fedcrg.artifacts.hashing import sha256_file
from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.references import CacheReference, CacheReferenceStore
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import CalibrationAssignmentMode, DatasetId, PolicyId
from fedcrg.core.ids import Sha256
from fedcrg.scoring.cache import ScoreCache


@dataclass(frozen=True, slots=True)
class FrozenCacheInputs:
    """Paths to immutable upstream evidence used by one policy cell."""

    prepared_root: Path
    model_path: Path
    training_manifest: Path
    score_root: Path


class PolicyCellMaterializer:
    """Reference reusable caches and materialize one auditable policy evaluation cell."""

    def __init__(
        self,
        evaluator: EvaluatePolicies | None = None,
        score_cache: ScoreCache | None = None,
        references: CacheReferenceStore | None = None,
    ) -> None:
        self.evaluator = evaluator or EvaluatePolicies()
        self.score_cache = score_cache or ScoreCache()
        self.references = references or CacheReferenceStore()

    def materialize(
        self,
        config: ExperimentConfig,
        policy: PolicyId,
        layout: RunLayout,
        caches: FrozenCacheInputs,
        calibration_seed: int,
        assignment_mode: CalibrationAssignmentMode = CalibrationAssignmentMode.SEEDED_PERMUTATION,
    ) -> object:
        self._validate_upstream(config, caches)
        self._copy_manifests(
            config,
            layout,
            caches,
            calibration_seed,
            assignment_mode,
        )
        self._write_cache_references(config, layout, caches)

        scores = self.score_cache.load(caches.score_root)
        if scores.data_spec_hash != Sha256(config.data_spec_hash):
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: data specification differs")
        if scores.training_spec_hash != Sha256(config.training_spec_hash):
            raise ValueError("SCORE_CACHE_HASH_MISMATCH: training specification differs")
        bundle = self.evaluator.evaluate(
            config,
            scores,
            calibration_seed=calibration_seed,
            mode=assignment_mode,
            prepared_root=caches.prepared_root,
        )
        self.evaluator.write_policy_artifacts(
            layout.root,
            layout.root.name,
            policy,
            bundle,
        )
        atomic_write_json(
            layout.reports / "evaluation_summary.json",
            {
                "calibration_seed": calibration_seed,
                "calibration_assignment": assignment_mode.value,
                "score_cache_sha256": scores.cache_sha256.value if scores.cache_sha256 else None,
                "evaluation": self.evaluator.to_serializable(bundle),
            },
        )
        return next(
            (item for item in bundle.federations if item.policy is policy),
            None,
        )

    @staticmethod
    def _validate_upstream(config: ExperimentConfig, caches: FrozenCacheInputs) -> None:
        required = (
            caches.prepared_root / "manifest.json",
            caches.prepared_root / "preprocessing.json",
            caches.model_path,
            caches.training_manifest,
            caches.score_root / ScoreCache.filename,
            caches.score_root / ScoreCache.manifest_filename,
        )
        missing = tuple(path for path in required if not path.is_file())
        if missing:
            raise FileNotFoundError(
                "Missing frozen upstream artifact(s): "
                + ", ".join(str(path) for path in missing)
            )
        prepared = json.loads(
            (caches.prepared_root / "manifest.json").read_text(encoding="utf-8")
        )
        training = json.loads(caches.training_manifest.read_text(encoding="utf-8"))
        score = json.loads(
            (caches.score_root / ScoreCache.manifest_filename).read_text(encoding="utf-8")
        )
        expected = (
            ("prepared dataset", prepared.get("data_spec_hash"), config.data_spec_hash),
            ("training", training.get("data_spec_hash"), config.data_spec_hash),
            ("training", training.get("training_spec_hash"), config.training_spec_hash),
            ("score cache", score.get("data_spec_hash"), config.data_spec_hash),
            ("score cache", score.get("training_spec_hash"), config.training_spec_hash),
        )
        for name, observed, wanted in expected:
            if observed != wanted:
                raise ValueError(f"{name} provenance hash does not match the requested cell")
        if training.get("model_file_sha256") != sha256_file(caches.model_path):
            raise ValueError("Frozen model hash does not match its training manifest")

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
        calibration_seed: int,
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
            "diad_eligibility.json"
            if config.dataset.id is DatasetId.DIAD
            else "eligibility.json"
        )
        eligibility_source = caches.prepared_root / eligibility_name
        if eligibility_source.exists():
            self._copy(eligibility_source, layout.data / eligibility_name)
        assignment_source = (
            caches.prepared_root / "splits" / "seeded" / f"c{calibration_seed}.json"
            if assignment_mode is CalibrationAssignmentMode.SEEDED_PERMUTATION
            else caches.prepared_root / "splits" / "source_order.json"
        )
        self._copy(assignment_source, layout.data / "calibration_assignment.json")
        self._copy(caches.training_manifest, layout.training / "training.json")
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
