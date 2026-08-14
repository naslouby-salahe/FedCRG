from fedcrg.config import Study
from fedcrg.types import DatasetId, DetectorId, ExperimentId, PolicyId


def _resolve(experiment_id: ExperimentId):
    return Study.load().resolve(experiment_id)


def test_primary_profile_is_frozen() -> None:
    config = _resolve(ExperimentId.PRIMARY_NBAIOT)
    assert config.id is ExperimentId.PRIMARY_NBAIOT
    assert config.dataset.id is DatasetId.NBAIOT
    assert config.detector is not None
    assert config.detector.id is DetectorId.AUTOENCODER
    assert config.detector.hidden_dims == (86, 57, 38, 29)
    assert config.training is not None
    assert config.training.rounds == 30
    assert config.training.local_epochs == 120
    assert config.randomness.model_seeds == (11, 22, 33, 44, 55)
    assert tuple(config.policies) == tuple(PolicyId)


def test_external_profile_is_frozen() -> None:
    config = _resolve(ExperimentId.EXTERNAL_DIAD)
    assert config.id is ExperimentId.EXTERNAL_DIAD
    assert config.dataset.id is DatasetId.DIAD
    assert config.dataset.expected_source_clients == 115
    assert config.detector is not None
    assert config.detector.hidden_dims == (64, 43, 28, 21)
    assert config.training is not None
    assert config.training.rounds == 30
    assert config.training.local_epochs == 20
    assert config.training.record_round20_score_correlation is True


def test_second_detector_profile_is_frozen() -> None:
    config = _resolve(ExperimentId.SECOND_DETECTOR)
    assert config.id is ExperimentId.SECOND_DETECTOR
    assert config.detector is not None
    assert config.detector.id is DetectorId.DEEP_SVDD
    assert config.detector.hidden_dims == (64,)
    assert config.detector.embedding_dim == 32
    assert config.randomness.model_seeds == (11, 22, 33)
    assert len(config.policies) == 5
