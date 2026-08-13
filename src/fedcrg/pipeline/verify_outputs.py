"""Run-level and repository-wide reproducibility verification."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.paths import RunLayout
from fedcrg.artifacts.manifests import RunManifestStore
from fedcrg.artifacts.integrity import ArtifactVerifier, VerificationResult
from fedcrg.domain.enums import ExperimentStatus
from fedcrg.experiments.completion import ExperimentCompletion, ExperimentCompletionAuditor
from fedcrg.experiments.experiment_definition import all_experiment_definitions
from fedcrg.method.mismatch_detection import minimum_bidirectional_sample_count
from fedcrg.method.calibration_readiness import ReadinessPlanCache


@dataclass(frozen=True, slots=True)
class PrecomputeVerification:
    valid: bool
    problems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunVerification:
    run_id: str
    result: VerificationResult


@dataclass(frozen=True, slots=True)
class RepositoryVerification:
    valid: bool
    runs: tuple[RunVerification, ...]
    experiment_completion: tuple[ExperimentCompletion, ...]
    test_return_code: int | None

    @property
    def incomplete_experiments(self) -> tuple[str, ...]:
        return tuple(
            row.experiment_id.value for row in self.experiment_completion if not row.complete
        )


class VerifyOutputs:
    """Verify artifact hashes, run semantics, complete workloads, and optionally tests."""

    def __init__(self) -> None:
        self.verifier = ArtifactVerifier()
        self.manifests = RunManifestStore()
        self.completion = ExperimentCompletionAuditor()

    @staticmethod
    def verify_protocol_precompute(outputs_root: Path) -> PrecomputeVerification:
        """Reconcile the frozen readiness/mismatch tables against their locked formulas."""

        problems: list[str] = []
        precomputed_root = outputs_root / "cache" / "precomputed"
        readiness_path = precomputed_root / "readiness_plans.json"
        mismatch_path = precomputed_root / "mismatch_cutoffs.json"

        if not readiness_path.is_file():
            problems.append(f"missing readiness table: {readiness_path}")
        else:
            try:
                cache = ReadinessPlanCache(readiness_path)
                if len(cache) == 0:
                    problems.append("readiness table is empty")
            except (ValueError, KeyError) as exc:
                problems.append(f"readiness table failed regeneration: {exc}")

        if not mismatch_path.is_file():
            problems.append(f"missing mismatch cutoff table: {mismatch_path}")
        else:
            try:
                payload = json.loads(mismatch_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                problems.append(f"mismatch cutoff table is not valid JSON: {exc}")
            else:
                cells = payload.get("cells")
                if not isinstance(cells, list) or not cells:
                    problems.append("mismatch cutoff table has no cells")
                lower = (
                    payload.get("band", {}).get("lower")
                    if isinstance(payload.get("band"), dict)
                    else None
                )
                confidence = payload.get("confidence")
                expected_minimum = payload.get("minimum_bidirectional_sample_count")
                if isinstance(lower, (int, float)) and isinstance(confidence, (int, float)):
                    regenerated_minimum = minimum_bidirectional_sample_count(
                        float(lower), float(confidence)
                    )
                    if regenerated_minimum != expected_minimum:
                        problems.append(
                            "mismatch cutoff table minimum sample count failed formula regeneration"
                        )

        return PrecomputeVerification(not problems, tuple(problems))

    def verify_run(self, run_root: Path) -> VerificationResult:
        layout = RunLayout(run_root)
        manifest = self.manifests.load(layout.manifest)
        if manifest.status is not ExperimentStatus.COMPLETE:
            return VerificationResult(False, ("manifest:not_complete",), (), ())
        if manifest.experiment_id not in {item.id for item in all_experiment_definitions()}:
            return VerificationResult(False, ("manifest:unknown_experiment",), (), ())
        return self.verifier.verify(layout)

    def verify_repository(
        self,
        outputs_root: Path,
        *,
        run_tests: bool = True,
        repository_root: Path = Path("."),
    ) -> RepositoryVerification:
        runs_root = outputs_root / "runs"
        run_results: list[RunVerification] = []
        if runs_root.exists():
            for run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
                try:
                    run_results.append(RunVerification(run_root.name, self.verify_run(run_root)))
                except Exception as exc:
                    run_results.append(
                        RunVerification(
                            run_root.name,
                            VerificationResult(
                                False,
                                (f"verification_exception:{type(exc).__name__}:{exc}",),
                                (),
                                (),
                            ),
                        )
                    )

        experiment_completion = self.completion.audit(outputs_root, repository_root)
        test_return_code: int | None = None
        if run_tests:
            process = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=repository_root,
                check=False,
            )
            test_return_code = process.returncode

        valid = (
            bool(run_results)
            and all(item.result.valid for item in run_results)
            and all(item.complete for item in experiment_completion)
            and (test_return_code in {None, 0})
        )
        return RepositoryVerification(
            valid,
            tuple(run_results),
            experiment_completion,
            test_return_code,
        )
