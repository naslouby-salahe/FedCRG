from pathlib import Path

import pytest

from fedcrg.pipeline.run_experiment import RunExperiment
from fedcrg.artifacts.manifests import RunManifestStore
from fedcrg.config.dataset_config import DatasetConfig, SplitConfig
from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.config.method_config import ProtocolConfig
from fedcrg.config.training_config import AutoencoderConfig, RandomnessConfig, TrainingConfig
from fedcrg.domain.enums import (
    DatasetFeatureContractId,
    DatasetId,
    ExperimentId,
    ExperimentStatus,
    PolicyId,
)
from fedcrg.experiments.execution import ExperimentPlan


def _config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        id=ExperimentId.COMPUTATIONAL_BENCHMARK,
        protocol=ProtocolConfig(),
        dataset=DatasetConfig(
            id=DatasetId.SYNTHETIC,
            feature_contract=DatasetFeatureContractId.SYNTHETIC,
            source_version="1",
            feature_count=2,
            minimum_clients=1,
            split=SplitConfig(
                train_benign=1,
                reference_benign=1,
                mismatch_benign=1,
                calibration_benign=1,
                benign_guard=0,
                min_benign_test=1,
                attack_dev=1,
                min_attack_test=1,
                min_attack_test_per_group=1,
            ),
            calibration_seeds=(1000,),
            primary_calibration_seed=1000,
        ),
        detector=AutoencoderConfig(hidden_dims=(2,)),
        training=TrainingConfig(rounds=1, local_epochs=1, batch_size=1),
        randomness=RandomnessConfig(model_seeds=(11,)),
        policies=(PolicyId.FEDCRG,),
        outputs_root=root,
    )


def test_run_application_creates_immutable_complete_run(tmp_path: Path) -> None:
    result, layout = RunExperiment().execute(
        ExperimentId.COMPUTATIONAL_BENCHMARK,
        _config(tmp_path),
        11,
        1000,
        PolicyId.FEDCRG,
        lambda plan, run_layout: {"done": True},
    )
    assert result == {"done": True}
    assert layout.manifest.exists()
    assert (layout.verification / "hashes.json").exists()


def test_run_application_records_failed_status(tmp_path: Path) -> None:
    config = _config(tmp_path)
    service = RunExperiment()

    def fail(plan: ExperimentPlan, run_layout: object) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        service.execute(
            ExperimentId.COMPUTATIONAL_BENCHMARK, config, 11, 1000, PolicyId.FEDCRG, fail
        )
    runs = list((tmp_path / "runs").iterdir())
    assert RunManifestStore().load(runs[0] / "manifest.json").status is ExperimentStatus.FAILED
