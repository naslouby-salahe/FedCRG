from fedcrg.config import Study
from fedcrg.types import ExperimentId


def test_catalogue_is_exactly_twenty_pre_registered_experiments() -> None:
    specs = Study.load().catalogue.all()
    ids = {spec.id for spec in specs}
    assert len(specs) == 20
    assert ids == set(ExperimentId)


def test_locked_synthetic_workload_counts() -> None:
    catalogue = Study.load().catalogue
    assert catalogue.spec(ExperimentId.READINESS_THEOREM).workload.monte_carlo_trials == 320_000
    assert catalogue.spec(ExperimentId.TARGET_FPR_SYNTHETIC).workload.monte_carlo_trials == 360_000
    assert catalogue.spec(ExperimentId.TEMPORAL_DEPENDENCE).workload.monte_carlo_trials == 120_000
    assert catalogue.spec(ExperimentId.CALIBRATION_SHIFT).workload.monte_carlo_trials == 50_000
    assert (
        catalogue.spec(ExperimentId.CALIBRATION_CONTAMINATION).workload.monte_carlo_trials
        == 120_000
    )
    assert catalogue.spec(ExperimentId.MISMATCH_POWER).workload.exact_cells == 45
