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
from fedcrg.experiments.completion import ExperimentCompletion, ExperimentCompletionAuditor
from fedcrg.experiments.registry import ExperimentRegistry


@dataclass(frozen=True, slots=True)
class RepositoryVerification:
    valid: bool
    runs: dict[str, VerificationResult]
    experiment_completion: tuple[ExperimentCompletion, ...]
    test_return_code: int | None

    @property
    def incomplete_experiments(self) -> tuple[str, ...]:
        return tuple(row.protocol_code for row in self.experiment_completion if not row.complete)


class VerifyOutputs:
    """Verify artifact hashes, run semantics, complete workloads, and optionally tests."""

    def __init__(self) -> None:
        self.verifier = ArtifactVerifier()
        self.manifests = RunManifestStore()
        self.registry = ExperimentRegistry()
        self.completion = ExperimentCompletionAuditor()

    def verify_run(self, run_root: Path) -> VerificationResult:
        layout = RunLayout(run_root)
        manifest = self.manifests.load(layout.manifest)
        if manifest.status is not ExperimentStatus.COMPLETE:
            return VerificationResult(False, ("manifest:not_complete",), (), {})
        if manifest.experiment_id not in {item.id for item in self.registry.all()}:
            return VerificationResult(False, ("manifest:unknown_experiment",), (), {})
        return self.verifier.verify(layout)

    def verify_repository(
        self,
        outputs_root: Path,
        *,
        run_tests: bool = True,
        repository_root: Path = Path("."),
    ) -> RepositoryVerification:
        runs_root = outputs_root / "runs"
        run_results: dict[str, VerificationResult] = {}
        if runs_root.exists():
            for run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
                try:
                    run_results[run_root.name] = self.verify_run(run_root)
                except Exception as exc:
                    run_results[run_root.name] = VerificationResult(
                        False,
                        (f"verification_exception:{type(exc).__name__}:{exc}",),
                        (),
                        {},
                    )

        experiment_completion = self.completion.audit(outputs_root)
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
            and all(item.complete for item in experiment_completion)
            and (test_return_code in {None, 0})
        )
        return RepositoryVerification(
            valid,
            run_results,
            experiment_completion,
            test_return_code,
        )
