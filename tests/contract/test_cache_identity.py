from fedcrg.config import Study
from fedcrg.types import ExperimentId


def _primary_config():
    return Study.load().resolve(ExperimentId.PRIMARY_NBAIOT)


def test_protocol_sensitivity_does_not_change_data_or_training_cache_identity() -> None:
    base = _primary_config()
    protocol = base.protocol.model_copy(update={"rho": 0.25})
    sensitivity = base.model_copy(update={"protocol": protocol})
    assert sensitivity.config_hash != base.config_hash
    assert sensitivity.data_spec_hash == base.data_spec_hash
    assert sensitivity.training_spec_hash == base.training_spec_hash


def test_calibration_seed_registry_does_not_enter_data_spec_hash() -> None:
    base = _primary_config()
    dataset = base.dataset.model_copy(
        update={"calibration_seeds": (1000,), "primary_calibration_seed": 1000}
    )
    named_only = base.model_copy(update={"dataset": dataset})
    assert named_only.config_hash != base.config_hash
    assert named_only.data_spec_hash == base.data_spec_hash
    assert named_only.training_spec_hash == base.training_spec_hash
