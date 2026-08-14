from __future__ import annotations

from fedcrg.config import Study, validate_experiment_config
from fedcrg.types import DatasetId, DetectorId, ExperimentId, PolicyId


def test_primary_config_resolves_without_filename_inference() -> None:
    config = Study.load().resolve(ExperimentId.PRIMARY_NBAIOT)
    validate_experiment_config(config)
    assert config.dataset.id is DatasetId.NBAIOT
    assert config.detector is not None
    assert config.detector.id is DetectorId.AUTOENCODER
    assert PolicyId.FEDCRG in config.policies
    assert len(config.config_hash) == 64
