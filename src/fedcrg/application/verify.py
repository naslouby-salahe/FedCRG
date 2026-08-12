"""Read-only verification of completed run evidence."""

from pathlib import Path

from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.manifest import RunManifestStore
from fedcrg.artifacts.verification import ArtifactVerifier, VerificationResult
from fedcrg.core.enums import ExperimentStatus


class VerifyOutputs:
    def __init__(self, verifier: ArtifactVerifier | None = None) -> None:
        self.verifier = verifier or ArtifactVerifier()
        self.manifests = RunManifestStore()

    def verify_run(self, run_root: Path) -> VerificationResult:
        layout = RunLayout(run_root)
        manifest = self.manifests.load(layout.manifest)
        if manifest.status is not ExperimentStatus.COMPLETE:
            return VerificationResult(False, ("manifest:not_complete",), (), {})
        return self.verifier.verify(layout)
