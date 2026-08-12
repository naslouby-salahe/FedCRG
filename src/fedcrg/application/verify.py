"""Run-level and repository-wide reproducibility verification."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.manifest import RunManifestStore
from fedcrg.artifacts.verification import ArtifactVerifier, VerificationResult
from fedcrg.core.enums import ExperimentStatus
from fedcrg.experiments.registry import ExperimentRegistry


@dataclass(frozen=True, slots=True)
class RepositoryVerification:
    valid: bool
    runs: dict[str, VerificationResult]
    missing_experiments: tuple[str, ...]
    test_return_code: int | None


class VerifyOutputs:
    def __init__(self) -> None:
        self.verifier = ArtifactVerifier()
        self.manifests = RunManifestStore()
        self.registry = ExperimentRegistry()

    def verify_run(self, run_root: Path) -> VerificationResult:
        layout = RunLayout(run_root)
        manifest = self.manifests.load(layout.manifest)
        if manifest.status is not ExperimentStatus.COMPLETE:
            return VerificationResult(False, ("manifest:not_complete",), (), {})
        result = self.verifier.verify(layout)
        if manifest.experiment_id not in {item.id for item in self.registry.all()}:
            return VerificationResult(False, ("manifest:unknown_experiment",), result.mismatched, result.hashes)
        return result

    def verify_repository(
        self,
        outputs_root: Path,
        *,
        run_tests: bool = True,
        repository_root: Path = Path("."),
    ) -> RepositoryVerification:
        runs_root = outputs_root / "runs"
        run_results: dict[str, VerificationResult] = {}
        completed_experiments: set[str] = set()
        if runs_root.exists():
            for run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
                result = self.verify_run(run_root)
                run_results[run_root.name] = result
                if result.valid:
                    completed_experiments.add(self.manifests.load(RunLayout(run_root).manifest).experiment_id.value)
        missing_experiments = tuple(
            item.protocol_code
            for item in self.registry.all()
            if item.id.value not in completed_experiments
        )
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
            and all(item.valid for item in run_results.values())
            and not missing_experiments
            and (test_return_code in {None, 0})
        )
        return RepositoryVerification(valid, run_results, missing_experiments, test_return_code)
