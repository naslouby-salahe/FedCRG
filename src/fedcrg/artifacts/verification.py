"""Artifact-ledger verification driven by the frozen experiment definition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.hashing import sha256_file
from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.artifacts.references import CacheReferenceStore
from fedcrg.core.enums import ArtifactType
from fedcrg.experiments.models import ExperimentDefinition


@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool
    missing: tuple[str, ...]
    mismatched: tuple[str, ...]
    hashes: dict[str, str]


class ArtifactVerifier:
    def _path_for(self, layout: RunLayout, artifact: ArtifactType) -> Path | None:
        mapping = {
            ArtifactType.RESOLVED_CONFIG: layout.resolved_config,
            ArtifactType.DATASET_MANIFEST: layout.data / "dataset_manifest.json",
            ArtifactType.ELIGIBILITY_MANIFEST: layout.data / "diad_eligibility.json",
            ArtifactType.SPLIT_MANIFEST: layout.data / "split_manifest.json",
            ArtifactType.PREPROCESSING_MANIFEST: layout.data / "preprocessing.json",
            ArtifactType.TRAINING_MANIFEST: layout.training / "training.json",
            ArtifactType.MODEL: layout.model_reference,
            ArtifactType.SCORE_MANIFEST: layout.scores / "manifest.json",
            ArtifactType.THRESHOLD_RECORDS: layout.threshold_records,
            ArtifactType.METRICS: layout.metric_records,
            ArtifactType.VERIFICATION: layout.verification / "hashes.json",
        }
        return mapping.get(artifact)

    def required_files(self, layout: RunLayout, definition: ExperimentDefinition) -> tuple[Path, ...]:
        required = [layout.run_config, layout.resolved_config, layout.environment]
        for artifact in definition.required_artifacts:
            path = self._path_for(layout, artifact)
            if path is not None and artifact is not ArtifactType.VERIFICATION:
                required.append(path)
        return tuple(dict.fromkeys(required))

    def _hashable_files(self, layout: RunLayout) -> tuple[Path, ...]:
        return tuple(
            sorted(
                (
                    path for path in layout.root.rglob("*")
                    if path.is_file()
                    and path != layout.manifest
                    and layout.verification not in path.parents
                ),
                key=lambda path: str(path.relative_to(layout.root)),
            )
        )

    def record(self, layout: RunLayout, definition: ExperimentDefinition) -> VerificationResult:
        missing = tuple(
            str(path.relative_to(layout.root))
            for path in self.required_files(layout, definition)
            if not path.exists()
        )
        hashes = {
            str(path.relative_to(layout.root)): sha256_file(path)
            for path in self._hashable_files(layout)
        }
        outputs_root = layout.root.parents[1]
        references = CacheReferenceStore()
        reference_mismatches: list[str] = []
        for reference_path in (layout.model_reference, layout.score_reference):
            if reference_path.exists():
                try:
                    if not references.load(reference_path).verify(outputs_root):
                        reference_mismatches.append(str(reference_path.relative_to(layout.root)))
                except Exception:
                    reference_mismatches.append(str(reference_path.relative_to(layout.root)))
        mismatched = tuple(sorted(reference_mismatches))
        result = VerificationResult(not missing and not mismatched, missing, mismatched, hashes)
        atomic_write_json(
            layout.verification / "hashes.json",
            {"missing": list(missing), "mismatched": list(mismatched), "hashes": hashes},
        )
        return result

    def verify(self, layout: RunLayout) -> VerificationResult:
        evidence_path = layout.verification / "hashes.json"
        if not evidence_path.exists():
            return VerificationResult(False, ("verification/hashes.json",), (), {})
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        expected: dict[str, str] = evidence.get("hashes", {})
        missing = tuple(sorted(path for path in expected if not (layout.root / path).exists()))
        local_mismatched = {
            path for path, expected_hash in expected.items()
            if (layout.root / path).exists() and sha256_file(layout.root / path) != expected_hash
        }
        outputs_root = layout.root.parents[1]
        references = CacheReferenceStore()
        for reference_path in (layout.model_reference, layout.score_reference):
            if reference_path.exists():
                try:
                    if not references.load(reference_path).verify(outputs_root):
                        local_mismatched.add(str(reference_path.relative_to(layout.root)))
                except Exception:
                    local_mismatched.add(str(reference_path.relative_to(layout.root)))
        mismatched = tuple(sorted(local_mismatched))
        originally_missing = tuple(str(item) for item in evidence.get("missing", []))
        all_missing = tuple(sorted(set(missing) | set(originally_missing)))
        return VerificationResult(not all_missing and not mismatched, all_missing, mismatched, expected)
