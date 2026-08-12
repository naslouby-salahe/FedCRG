from pathlib import Path

from fedcrg.application.run_experiment import RunExperiment
from fedcrg.config.models import AutoencoderConfig, DatasetConfig, ExperimentConfig, ProtocolConfig, RandomnessConfig, SplitConfig, TrainingConfig
from fedcrg.core.enums import DatasetId, ExperimentId, PolicyId


def _config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        id=ExperimentId.PRIMARY_NBAIOT.value,
        protocol=ProtocolConfig(),
        dataset=DatasetConfig(id=DatasetId.NBAIOT, feature_count=115, expected_clients=1, minimum_clients=1, split=SplitConfig(train_benign=1, reference_benign=1, mismatch_benign=736, calibration_benign=1416, benign_guard=0, min_benign_test=1, attack_dev=1, min_attack_test=1, min_attack_test_per_group=1), calibration_seeds=(1000,), primary_calibration_seed=1000),
        detector=AutoencoderConfig(hidden_dims=(2,)),
        training=TrainingConfig(rounds=1, local_epochs=1, batch_size=1),
        randomness=RandomnessConfig(model_seeds=(11,)),
        policies=(PolicyId.FEDCRG,),
        outputs_root=root,
    )


def test_run_application_creates_immutable_complete_run(tmp_path: Path) -> None:
    result, layout = RunExperiment().execute(ExperimentId.PRIMARY_NBAIOT, _config(tmp_path), 11, 1000, lambda plan: {"done": True})
    assert result == {"done": True}
    assert layout.manifest.exists()
    assert (layout.verification / "hashes.json").exists()


def test_run_application_records_failed_status(tmp_path: Path) -> None:
    import pytest
    from fedcrg.artifacts.manifest import RunManifestStore
    from fedcrg.core.enums import ExperimentStatus
    config = _config(tmp_path)
    service = RunExperiment()
    def fail(plan): raise RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        service.execute(ExperimentId.PRIMARY_NBAIOT, config, 11, 1000, fail)
    runs = list((tmp_path / "runs").iterdir())
    assert RunManifestStore().load(runs[0] / "manifest.json").status is ExperimentStatus.FAILED
