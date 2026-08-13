from pathlib import Path
from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.manifest import RunManifest, RunManifestStore
from fedcrg.artifacts.serialization import atomic_write_json, atomic_write_text
from fedcrg.artifacts.verification import ArtifactVerifier
from fedcrg.domain.enums import (
    ExperimentCode,
    ExperimentId,
    ExperimentStatus,
    ExperimentType,
    PolicyId,
)
from fedcrg.domain.identifiers import CalibrationSeed, ModelSeed, RunId, Sha256
from fedcrg.experiments.experiment_definition import ExperimentDefinition


def test_artifact_verifier_hashes_run_files(tmp_path: Path) -> None:
    layout = RunLayout.for_run(tmp_path, RunId("r1"))
    layout.create()
    RunManifestStore().save(
        layout.manifest,
        RunManifest(
            run_id=RunId("r1"),
            experiment_id=ExperimentId.PRIMARY_NBAIOT,
            policy_id=PolicyId.FEDCRG,
            config_hash=Sha256("a" * 64),
            model_seed=ModelSeed(11),
            calibration_seed=CalibrationSeed(1000),
            status=ExperimentStatus.RUNNING,
        ),
    )
    atomic_write_text(layout.resolved_config, "x: 1\n")
    atomic_write_json(layout.environment, {"python": "test"})
    atomic_write_json(layout.run_config, {"run_id": "r1"})
    verifier = ArtifactVerifier()
    definition = ExperimentDefinition(
        id=ExperimentId.PRIMARY_NBAIOT,
        protocol_code=ExperimentCode.R1,
        type=ExperimentType.PRIMARY,
    )
    recorded = verifier.record(layout, definition)
    assert recorded.valid
    result = verifier.verify(layout)
    assert result.valid
    assert result.hash_for("resolved_config.yaml") is not None
    assert result.hash_for("manifest.json") is None
    assert (layout.verification / "hashes.json").exists()
