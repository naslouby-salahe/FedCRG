from pathlib import Path
from fedcrg.config.resolve import ExperimentConfigResolver
from fedcrg.config.validate import validate_experiment_config
from fedcrg.domain.enums import DatasetId, DetectorId, PolicyId


def test_primary_config_resolves_without_filename_inference() -> None:
    config = ExperimentConfigResolver().resolve(Path("configs/experiments/primary/nbaiot.yaml"))
    validate_experiment_config(config)
    assert config.dataset.id is DatasetId.NBAIOT
    assert config.detector.id is DetectorId.AUTOENCODER
    assert PolicyId.FEDCRG in config.policies
    assert len(config.config_hash) == 64
