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
    for path in (
        layout.dataset_manifest,
        layout.preprocessing_evidence,
        layout.training_manifest,
        layout.model_reference,
        layout.score_manifest,
        layout.threshold_records,
        layout.metric_records,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    verifier = ArtifactVerifier()
    definition = Study.load().spec(ExperimentId.READINESS_SAMPLE_SIZE)
    recorded = verifier.record(layout, definition)
    assert recorded.valid
    assert recorded.hash_for("resolved_config.yaml") is not None
    assert recorded.hash_for("manifest.json") is not None
    assert (layout.verification / "hashes.json").exists()


def test_artifact_verifier_checks_cache_pointer_against_target(tmp_path: Path) -> None:
    outputs = tmp_path
    layout = OutputsLayout(outputs).run("r2")
    layout.create()
    target = outputs / "cached-model"
    target.write_bytes(b"model-bytes")
    from fedcrg.evidence.models import CacheReference
    from fedcrg.hashing import sha256_file

    layout.model_reference.write_text(
        CacheReference(
            relative_path="cached-model",
            sha256=sha256_file(target),
            size_bytes=target.stat().st_size,
        ).model_dump_json(),
        encoding="utf-8",
    )
    layout.run_config.write_text("{}\n", encoding="utf-8")
    layout.resolved_config.write_text("x: 1\n", encoding="utf-8")
    layout.environment.write_text("{}\n", encoding="utf-8")
    layout.manifest.write_text("{}\n", encoding="utf-8")
    verifier = ArtifactVerifier()
    definition = Study.load().spec(ExperimentId.READINESS_THEOREM)
    recorded = verifier.record(layout, definition)
    assert recorded.valid

    target.write_bytes(b"stale")
    recorded = verifier.record(layout, definition)
    assert not recorded.valid
    assert recorded.mismatched
