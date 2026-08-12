from pathlib import Path
from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.manifest import RunManifest, RunManifestStore
from fedcrg.artifacts.serialization import atomic_write_json, atomic_write_text
from fedcrg.artifacts.verification import ArtifactVerifier
from fedcrg.core.enums import ExperimentStatus
from fedcrg.core.ids import RunId

def test_artifact_verifier_hashes_run_files(tmp_path: Path) -> None:
    layout=RunLayout.for_run(tmp_path,RunId("r1")); layout.create(); RunManifestStore().save(layout.manifest,RunManifest("r1","primary_nbaiot","hash",11,1000,ExperimentStatus.RUNNING)); atomic_write_text(layout.resolved_config,"x: 1\n"); atomic_write_json(layout.environment,{"python":"test"}); verifier=ArtifactVerifier(); recorded=verifier.record(layout); assert recorded.valid; result=verifier.verify(layout); assert result.valid; assert "resolved_config.yaml" in result.hashes; assert "manifest.json" not in result.hashes; assert (layout.verification/"hashes.json").exists()
