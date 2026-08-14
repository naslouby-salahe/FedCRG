from __future__ import annotations

from pathlib import Path

import pytest

from fedcrg.config import (
    ExperimentSpec,
    ParameterAxis,
    ParameterCell,
    Study,
    WorkloadExpectation,
)
from fedcrg.experiments.analyses import (
    RunSyntheticExperiments,
    SyntheticExperimentEnvelope,
)
from fedcrg.types import (
    ContaminationDirection,
    DatasetId,
    ExperimentAxisId,
    ExperimentId,
    ExperimentType,
    SyntheticDistribution,
)


@pytest.fixture(scope="module")
def base_config():
    return Study.load().resolve(ExperimentId.READINESS_THEOREM)


def _spec(
    experiment_id: ExperimentId,
    axes: dict[ExperimentAxisId, tuple[object, ...]],
    *,
    coupled_cells: tuple[dict[str, object], ...] = (),
    category: ExperimentType = ExperimentType.SYNTHETIC,
    workload: WorkloadExpectation | None = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        id=experiment_id,
        category=category,
        dataset=DatasetId.SYNTHETIC,
        axes=[ParameterAxis(id=axis_id, values=list(values)) for axis_id, values in axes.items()],
        coupled_cells=[ParameterCell.model_validate(cell) for cell in coupled_cells],
        workload=workload or WorkloadExpectation(),
    )


def test_int_values_coerces_and_validates() -> None:
    spec = _spec(ExperimentId.READINESS_THEOREM, {ExperimentAxisId.CALIBRATION_N: (10, 20, 30)})
    assert RunSyntheticExperiments._int_values(spec, ExperimentAxisId.CALIBRATION_N) == (
        10,
        20,
        30,
    )


def test_int_values_rejects_non_integers() -> None:
    spec = _spec(ExperimentId.READINESS_THEOREM, {ExperimentAxisId.CALIBRATION_N: (1.5,)})
    with pytest.raises(TypeError):
        RunSyntheticExperiments._int_values(spec, ExperimentAxisId.CALIBRATION_N)


def test_float_values_coerces_ints_and_floats() -> None:
    spec = _spec(ExperimentId.CALIBRATION_SHIFT, {ExperimentAxisId.MEAN_SHIFT: (0, 0.5, 1)})
    assert RunSyntheticExperiments._float_values(spec, ExperimentAxisId.MEAN_SHIFT) == (
        0.0,
        0.5,
        1.0,
    )


def test_float_values_rejects_non_numeric_axis_values() -> None:
    spec = _spec(
        ExperimentId.CALIBRATION_SHIFT,
        {ExperimentAxisId.MEAN_SHIFT: (SyntheticDistribution.NORMAL,)},
    )
    with pytest.raises(TypeError):
        RunSyntheticExperiments._float_values(spec, ExperimentAxisId.MEAN_SHIFT)


def test_distributions_returns_declared_axis_values() -> None:
    spec = _spec(
        ExperimentId.READINESS_THEOREM,
        {ExperimentAxisId.DISTRIBUTION: (SyntheticDistribution.NORMAL,)},
    )
    assert RunSyntheticExperiments._distributions(spec) == (SyntheticDistribution.NORMAL,)


def test_distributions_rejects_a_malformed_axis() -> None:
    spec = _spec(ExperimentId.READINESS_THEOREM, {ExperimentAxisId.CALIBRATION_N: (1,)})
    malformed_axis = ParameterAxis.model_construct(
        id=ExperimentAxisId.DISTRIBUTION, values=(ContaminationDirection.HIGH,)
    )
    spec = spec.model_copy(update={"axes": (*spec.axes, malformed_axis)})
    with pytest.raises(TypeError):
        RunSyntheticExperiments._distributions(spec)


def test_directions_returns_declared_axis_values() -> None:
    spec = _spec(
        ExperimentId.CALIBRATION_CONTAMINATION,
        {ExperimentAxisId.DIRECTION: (ContaminationDirection.HIGH, ContaminationDirection.LOW)},
    )
    assert RunSyntheticExperiments._directions(spec) == (
        ContaminationDirection.HIGH,
        ContaminationDirection.LOW,
    )


def test_directions_rejects_a_malformed_axis() -> None:
    spec = _spec(ExperimentId.CALIBRATION_CONTAMINATION, {ExperimentAxisId.FRACTION: (0.1,)})
    malformed_axis = ParameterAxis.model_construct(
        id=ExperimentAxisId.DIRECTION, values=(SyntheticDistribution.NORMAL,)
    )
    spec = spec.model_copy(update={"axes": (*spec.axes, malformed_axis)})
    with pytest.raises(TypeError):
        RunSyntheticExperiments._directions(spec)


def test_write_persists_a_readable_envelope(tmp_path: Path, base_config) -> None:
    spec = _spec(
        ExperimentId.READINESS_THEOREM,
        {ExperimentAxisId.CALIBRATION_N: (10,)},
    )
    output = tmp_path / "envelope.json"
    RunSyntheticExperiments().write(output, spec, (), 0)
    envelope = SyntheticExperimentEnvelope.model_validate_json(output.read_text())
    assert envelope.experiment_id is ExperimentId.READINESS_THEOREM
    assert envelope.actual_cells == 0


def test_write_rejects_trial_ledger_mismatch(tmp_path: Path) -> None:
    spec = _spec(
        ExperimentId.READINESS_THEOREM,
        {ExperimentAxisId.CALIBRATION_N: (10,)},
        workload=WorkloadExpectation(monte_carlo_trials=999),
    )
    with pytest.raises(RuntimeError):
        RunSyntheticExperiments().write(tmp_path / "out.json", spec, (), 1)


def test_write_rejects_cell_ledger_mismatch(tmp_path: Path) -> None:
    spec = _spec(
        ExperimentId.MISMATCH_POWER,
        {ExperimentAxisId.MISMATCH_N: (10,), ExperimentAxisId.TRUE_FPR: (0.01,)},
        workload=WorkloadExpectation(exact_cells=5),
    )
    with pytest.raises(RuntimeError):
        RunSyntheticExperiments().write(tmp_path / "out.json", spec, (), 0)


def test_run_rejects_unsupported_experiment_id(tmp_path: Path, base_config) -> None:
    spec = _spec(ExperimentId.PRIMARY_NBAIOT, {}, category=ExperimentType.PRIMARY)
    with pytest.raises(ValueError):
        RunSyntheticExperiments().run(
            ExperimentId.PRIMARY_NBAIOT, spec, base_config, tmp_path / "out.json"
        )


def test_run_readiness_theorem_produces_one_cell_per_distribution_and_sample_size(
    tmp_path: Path, base_config
) -> None:
    spec = _spec(
        ExperimentId.READINESS_THEOREM,
        {
            ExperimentAxisId.DISTRIBUTION: (SyntheticDistribution.NORMAL,),
            ExperimentAxisId.CALIBRATION_N: (50, 60),
            ExperimentAxisId.REPETITIONS: (5,),
        },
    )
    output = RunSyntheticExperiments().run(
        ExperimentId.READINESS_THEOREM, spec, base_config, tmp_path / "readiness.json"
    )
    envelope = SyntheticExperimentEnvelope.model_validate_json(output.read_text())
    assert envelope.actual_cells == 2
    assert envelope.actual_monte_carlo_trials == 10


def test_run_target_fpr_synthetic_produces_one_cell_per_distribution_and_coupled_cell(
    tmp_path: Path, base_config
) -> None:
    spec = _spec(
        ExperimentId.TARGET_FPR_SYNTHETIC,
        {
            ExperimentAxisId.DISTRIBUTION: (SyntheticDistribution.NORMAL,),
            ExperimentAxisId.REPETITIONS: (5,),
        },
        coupled_cells=({"alpha": 0.05, "calibration_n": 40},),
    )
    output = RunSyntheticExperiments().run(
        ExperimentId.TARGET_FPR_SYNTHETIC, spec, base_config, tmp_path / "target_fpr.json"
    )
    envelope = SyntheticExperimentEnvelope.model_validate_json(output.read_text())
    assert envelope.actual_cells == 1
    assert envelope.actual_monte_carlo_trials == 5


def test_run_temporal_dependence_produces_one_cell_per_phi_and_sample_size(
    tmp_path: Path, base_config
) -> None:
    spec = _spec(
        ExperimentId.TEMPORAL_DEPENDENCE,
        {
            ExperimentAxisId.PHI: (0.1, 0.2),
            ExperimentAxisId.CALIBRATION_N: (40,),
            ExperimentAxisId.REPETITIONS: (5,),
        },
        category=ExperimentType.ROBUSTNESS,
    )
    output = RunSyntheticExperiments().run(
        ExperimentId.TEMPORAL_DEPENDENCE, spec, base_config, tmp_path / "temporal.json"
    )
    envelope = SyntheticExperimentEnvelope.model_validate_json(output.read_text())
    assert envelope.actual_cells == 2
    assert envelope.actual_monte_carlo_trials == 10


def test_run_calibration_shift_produces_one_cell_per_mean_shift(
    tmp_path: Path, base_config
) -> None:
    spec = _spec(
        ExperimentId.CALIBRATION_SHIFT,
        {
            ExperimentAxisId.MEAN_SHIFT: (0.0, 0.5),
            ExperimentAxisId.REPETITIONS: (5,),
        },
        category=ExperimentType.ROBUSTNESS,
    )
    output = RunSyntheticExperiments().run(
        ExperimentId.CALIBRATION_SHIFT, spec, base_config, tmp_path / "shift.json"
    )
    envelope = SyntheticExperimentEnvelope.model_validate_json(output.read_text())
    assert envelope.actual_cells == 2
    assert envelope.actual_monte_carlo_trials == 10


def test_run_calibration_contamination_produces_one_cell_per_fraction_and_direction(
    tmp_path: Path, base_config
) -> None:
    spec = _spec(
        ExperimentId.CALIBRATION_CONTAMINATION,
        {
            ExperimentAxisId.FRACTION: (0.0, 0.01),
            ExperimentAxisId.DIRECTION: (ContaminationDirection.HIGH, ContaminationDirection.LOW),
            ExperimentAxisId.REPETITIONS: (5,),
        },
        category=ExperimentType.ROBUSTNESS,
    )
    output = RunSyntheticExperiments().run(
        ExperimentId.CALIBRATION_CONTAMINATION,
        spec,
        base_config,
        tmp_path / "contamination.json",
    )
    envelope = SyntheticExperimentEnvelope.model_validate_json(output.read_text())
    assert envelope.actual_cells == 4
    assert envelope.actual_monte_carlo_trials == 20


def test_run_mismatch_power_produces_one_cell_per_sample_count_and_true_fpr(
    tmp_path: Path, base_config
) -> None:
    spec = _spec(
        ExperimentId.MISMATCH_POWER,
        {
            ExperimentAxisId.MISMATCH_N: (10, 20),
            ExperimentAxisId.TRUE_FPR: (0.01, 0.02),
        },
    )
    output = RunSyntheticExperiments().run(
        ExperimentId.MISMATCH_POWER, spec, base_config, tmp_path / "mismatch.json"
    )
    envelope = SyntheticExperimentEnvelope.model_validate_json(output.read_text())
    assert envelope.actual_cells == 4
    assert envelope.actual_monte_carlo_trials == 0
