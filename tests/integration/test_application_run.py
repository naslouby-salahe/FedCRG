from pathlib import Path

import pytest

from fedcrg.pipeline.run_experiment import RunExperiment
from fedcrg.artifacts.manifests import RunManifestStore
from fedcrg.config.dataset_config import DatasetConfig, SplitConfig
from fedcrg.config.detector_config import AutoencoderConfig
from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.domain.enums import (
    ComputeDeviceId,
    DatasetFeatureContractId,
    DatasetId,
    ExperimentId,
    ExperimentStatus,
    PolicyId,
)
from fedcrg.experiments.execution import ExperimentPlan
from tests._fixtures import (
    primary_protocol,
    primary_randomness,
    primary_statistics,
    primary_training,
)


def _config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        id=ExperimentId.COMPUTATIONAL_BENCHMARK,
        protocol=primary_protocol(),
        dataset=DatasetConfig(
            id=DatasetId.SYNTHETIC,
            feature_contract=DatasetFeatureContractId.SYNTHETIC,
            source_version="1",
            parser_version="1",
            feature_count=2,
            minimum_clients=1,
            expected_benign_counts={},
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
        detector=AutoencoderConfig(hidden_dims=(2,), xavier_tanh_gain=5.0 / 3.0),
        training=primary_training().model_copy(
            update={
                "rounds": 1,
                "local_epochs": 1,
                "batch_size": 1,
                "device": ComputeDeviceId.CPU,
            }
        ),
        randomness=primary_randomness(),
        statistics=primary_statistics(),
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
