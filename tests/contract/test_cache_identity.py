from pathlib import Path

from fedcrg.config.resolver import ExperimentConfigResolver

ROOT = Path(__file__).resolve().parents[2]


def test_protocol_sensitivity_does_not_change_data_or_training_cache_identity() -> None:
    base = ExperimentConfigResolver().resolve(ROOT / "configs/experiments/primary/nbaiot.yaml")
    protocol = base.protocol.model_copy(update={"rho": 0.25})
    sensitivity = base.model_copy(update={"protocol": protocol})
    assert sensitivity.config_hash != base.config_hash
    assert sensitivity.data_spec_hash == base.data_spec_hash
    assert sensitivity.training_spec_hash == base.training_spec_hash


def test_calibration_seed_registry_does_not_enter_data_spec_hash() -> None:
    base = ExperimentConfigResolver().resolve(ROOT / "configs/experiments/primary/nbaiot.yaml")
    dataset = base.dataset.model_copy(
        update={"calibration_seeds": (1000,), "primary_calibration_seed": 1000}
    )
    named_only = base.model_copy(update={"dataset": dataset})
    assert named_only.config_hash != base.config_hash
    assert named_only.data_spec_hash == base.data_spec_hash
    assert named_only.training_spec_hash == base.training_spec_hash
