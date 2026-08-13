"""Cryptographic and semantic verification of immutable run evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.hashing import sha256_file
from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.manifest import RunManifestStore
from fedcrg.artifacts.references import CacheReferenceStore
from fedcrg.artifacts.serialization import as_json_float, as_json_int, atomic_write_json
from fedcrg.core.enums import ArtifactType
from fedcrg.experiments.models import ExperimentDefinition


@dataclass(frozen=True, slots=True)
class FileHashRecord:
    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool
    missing: tuple[str, ...]
    mismatched: tuple[str, ...]
    hashes: tuple[FileHashRecord, ...]

    def hash_for(self, relative_path: str) -> str | None:
        for record in self.hashes:
            if record.relative_path == relative_path:
                return record.sha256
        return None


class ArtifactVerifier:
    """Verify file hashes and the provenance chain across dataset/model/score/policy evidence."""

    def _path_for(self, layout: RunLayout, artifact: ArtifactType) -> Path | None:
        mapping = {
            ArtifactType.RESOLVED_CONFIG: layout.resolved_config,
            ArtifactType.DATASET_MANIFEST: layout.data / "dataset_manifest.json",
            ArtifactType.ELIGIBILITY_MANIFEST: layout.data / "eligibility.json",
            ArtifactType.SPLIT_MANIFEST: layout.data / "calibration_assignment.json",
            ArtifactType.PREPROCESSING_MANIFEST: layout.data / "preprocessing.json",
            ArtifactType.TRAINING_MANIFEST: layout.training / "training.json",
            ArtifactType.MODEL: layout.model_reference,
            ArtifactType.SCORE_MANIFEST: layout.scores / "manifest.json",
            ArtifactType.THRESHOLD_RECORDS: layout.threshold_records,
            ArtifactType.METRICS: layout.metric_records,
            ArtifactType.VERIFICATION: layout.verification / "hashes.json",
        }
        return mapping.get(artifact)

    def required_files(
        self,
        layout: RunLayout,
        definition: ExperimentDefinition,
    ) -> tuple[Path, ...]:
        required = [
            layout.run_config,
            layout.resolved_config,
            layout.environment,
            layout.manifest,
        ]
        for artifact in definition.required_artifacts:
            path = self._path_for(layout, artifact)
            if path is not None and artifact is not ArtifactType.VERIFICATION:
                required.append(path)
        return tuple(dict.fromkeys(required))

    def _hashable_files(self, layout: RunLayout) -> tuple[Path, ...]:
        return tuple(
            sorted(
                (
                    path
                    for path in layout.root.rglob("*")
                    if path.is_file()
                    and path != layout.manifest
                    and layout.verification not in path.parents
                ),
                key=lambda path: str(path.relative_to(layout.root)),
            )
        )

    def record(
        self,
        layout: RunLayout,
        definition: ExperimentDefinition,
    ) -> VerificationResult:
        missing = tuple(
            str(path.relative_to(layout.root))
            for path in self.required_files(layout, definition)
            if not path.exists()
        )
        hashes = tuple(
            FileHashRecord(str(path.relative_to(layout.root)), sha256_file(path))
            for path in self._hashable_files(layout)
        )
        mismatched = tuple(sorted(self._semantic_mismatches(layout)))
        result = VerificationResult(not missing and not mismatched, missing, mismatched, hashes)
        atomic_write_json(
            layout.verification / "hashes.json",
            {
                "missing": list(missing),
                "mismatched": list(mismatched),
                "hashes": {record.relative_path: record.sha256 for record in hashes},
            },
        )
        return result

    def verify(self, layout: RunLayout) -> VerificationResult:
        evidence_path = layout.verification / "hashes.json"
        if not evidence_path.exists():
            return VerificationResult(False, ("verification/hashes.json",), (), ())
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        expected_raw: dict[str, str] = evidence.get("hashes", {})
        expected = tuple(FileHashRecord(path, digest) for path, digest in expected_raw.items())
        missing = tuple(sorted(path for path in expected_raw if not (layout.root / path).exists()))
        mismatched = {
            record.relative_path
            for record in expected
            if (layout.root / record.relative_path).exists()
            and sha256_file(layout.root / record.relative_path) != record.sha256
        }
        mismatched.update(self._semantic_mismatches(layout))
        originally_missing = tuple(str(item) for item in evidence.get("missing", []))
        originally_mismatched = tuple(str(item) for item in evidence.get("mismatched", []))
        all_missing = tuple(sorted(set(missing) | set(originally_missing)))
        all_mismatched = tuple(sorted(set(mismatched) | set(originally_mismatched)))
        return VerificationResult(
            not all_missing and not all_mismatched,
            all_missing,
            all_mismatched,
            expected,
        )

    def _semantic_mismatches(self, layout: RunLayout) -> set[str]:
        mismatches: set[str] = set()
        outputs_root = layout.root.parents[1]
        references = CacheReferenceStore()
        for label, reference_path in (
            ("model_reference", layout.model_reference),
            ("score_reference", layout.score_reference),
        ):
            if reference_path.exists():
                try:
                    if not references.load(reference_path).verify(outputs_root):
                        mismatches.add(f"{label}:target_hash")
                except Exception:
                    mismatches.add(f"{label}:invalid")

        try:
            manifest = RunManifestStore().load(layout.manifest)
            run_config = self._json(layout.run_config)
            dataset = self._json(layout.data / "dataset_manifest.json")
            training = self._json(layout.training / "training.json")
            score = self._json(layout.scores / "manifest.json")
            assignment = self._json(layout.data / "calibration_assignment.json")

            if run_config.get("run_id") != manifest.run_id.value:
                mismatches.add("run_config:run_id")
            if run_config.get("policy_id") != manifest.policy_id.value:
                mismatches.add("run_config:policy_id")
            if run_config.get("config_hash") != manifest.config_hash.value:
                mismatches.add("run_config:config_hash")
            if as_json_int(assignment.get("calibration_seed", -1)) != manifest.calibration_seed:
                mismatches.add("calibration_assignment:seed")

            data_spec = dataset.get("data_spec_hash")
            training_spec = training.get("training_spec_hash")
            if training.get("data_spec_hash") != data_spec:
                mismatches.add("training:data_spec_hash")
            if score.get("data_spec_hash") != data_spec:
                mismatches.add("score:data_spec_hash")
            if score.get("training_spec_hash") != training_spec:
                mismatches.add("score:training_spec_hash")

            if layout.model_reference.exists():
                model_reference = references.load(layout.model_reference)
                if training.get("model_file_sha256") != model_reference.sha256.value:
                    mismatches.add("model_reference:manifest_hash")
            if layout.score_reference.exists():
                score_reference = references.load(layout.score_reference)
                if score.get("score_cache_sha256") != score_reference.sha256.value:
                    mismatches.add("score_reference:manifest_hash")

            self._validate_records(layout, manifest.policy_id.value, mismatches)
        except FileNotFoundError:
            pass
        except Exception as exc:
            mismatches.add(f"semantic_validation:{type(exc).__name__}")
        return mismatches

    @staticmethod
    def _json(path: Path) -> dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object: {path}")
        return payload

    @staticmethod
    def _validate_records(
        layout: RunLayout,
        policy_id: str,
        mismatches: set[str],
    ) -> None:
        threshold_rows = ArtifactVerifier._jsonl(layout.threshold_records)
        metric_rows = ArtifactVerifier._jsonl(layout.metric_records)
        if not threshold_rows or not metric_rows:
            mismatches.add("records:empty")
            return
        threshold_clients = set()
        metric_clients = set()
        for row in threshold_rows:
            if row.get("policy_id") != policy_id:
                mismatches.add("threshold_records:policy")
            threshold_clients.add(str(row.get("client_id")))
            for field in (
                "tau_ref",
                "selected_tau",
                "readiness_probability",
                "cp_lower",
                "cp_upper",
            ):
                value = row.get(field)
                if value is not None and not math.isfinite(as_json_float(value)):
                    mismatches.add(f"threshold_records:nonfinite:{field}")
        for row in metric_rows:
            if row.get("policy_id") != policy_id:
                mismatches.add("metric_records:policy")
            metric_clients.add(str(row.get("client_id")))
            for field in ("fpr", "auroc", "auprc", "band_error"):
                value = row.get(field)
                if value is None or not math.isfinite(as_json_float(value)):
                    mismatches.add(f"metric_records:nonfinite:{field}")
        if threshold_clients != metric_clients:
            mismatches.add("records:client_set")

    @staticmethod
    def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSONL object in {path}")
            rows.append(payload)
        return tuple(rows)
