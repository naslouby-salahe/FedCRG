from fedcrg.domain.enums import ExperimentId
from fedcrg.experiments.experiment_definition import all_experiment_definitions


def test_catalogue_is_exactly_twenty_pre_registered_experiments() -> None:
    rows = all_experiment_definitions()
    ids = {row.id for row in rows}
    assert len(rows) == 20
    assert ids == set(ExperimentId)


def test_locked_synthetic_workload_counts() -> None:
    by_id = {row.id: row for row in all_experiment_definitions()}
    assert by_id[ExperimentId.READINESS_THEOREM].workload.monte_carlo_trials == 320_000
    assert by_id[ExperimentId.TARGET_FPR_SYNTHETIC].workload.monte_carlo_trials == 360_000
    assert by_id[ExperimentId.TEMPORAL_DEPENDENCE].workload.monte_carlo_trials == 120_000
    assert by_id[ExperimentId.CALIBRATION_SHIFT].workload.monte_carlo_trials == 50_000
    assert by_id[ExperimentId.CALIBRATION_CONTAMINATION].workload.monte_carlo_trials == 120_000
    assert by_id[ExperimentId.MISMATCH_POWER].workload.exact_cells == 45
