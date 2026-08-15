"""Integration tests for hashing and verifying run artifacts."""

from __future__ import annotations

from pathlib import Path

from fedcrg.config import Study
from fedcrg.evidence.models import RunManifest
from fedcrg.evidence.store import (
    ArtifactVerifier,
    RunManifestStore,
    atomic_write_json,
    atomic_write_text,
)
from fedcrg.paths import OutputsLayout
from fedcrg.types import ExperimentId, ExperimentStatus, PolicyId


def test_artifact_verifier_hashes_run_files(tmp_path: Path) -> None:
    """The verifier hashes each expected run file and records the results as valid."""
    layout = OutputsLayout(tmp_path).run("r1")
    layout.create()
    RunManifestStore().save(
        layout.manifest,
        RunManifest(
            run_id="r1",
            experiment_id=ExperimentId.PRIMARY_NBAIOT,
            policy_id=PolicyId.FEDCRG,
            config_hash="a" * 64,
            model_seed=11,
            calibration_seed=1000,
            status=ExperimentStatus.RUNNING,
        ),
    )
    atomic_write_text(layout.resolved_config, "x: 1\n")
    atomic_write_json(layout.environment, {"python": "test"})
    atomic_write_json(layout.run_config, {"run_id": "r1"})
    verifier = ArtifactVerifier()
    definition = Study.load().spec(ExperimentId.READINESS_SAMPLE_SIZE)
    recorded = verifier.record(layout, definition)
    assert recorded.valid
    assert recorded.hash_for("resolved_config.yaml") is not None
    assert recorded.hash_for("manifest.json") is not None
    assert (layout.verification / "hashes.json").exists()
