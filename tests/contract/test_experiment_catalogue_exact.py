from fedcrg.core.enums import ExperimentCode
from fedcrg.experiments.registry import ExperimentRegistry


def test_catalogue_is_exactly_s1_to_s6_and_r1_to_r14() -> None:
    rows = ExperimentRegistry().all()
    codes = {row.protocol_code for row in rows}
    assert len(rows) == 20
    assert codes == {ExperimentCode(f"S{i}") for i in range(1, 7)} | {
        ExperimentCode(f"R{i}") for i in range(1, 15)
    }


def test_locked_synthetic_workload_counts() -> None:
    by_code = {row.protocol_code: row for row in ExperimentRegistry().all()}
    assert by_code[ExperimentCode.S1].workload.monte_carlo_trials == 320_000
    assert by_code[ExperimentCode.S2].workload.monte_carlo_trials == 360_000
    assert by_code[ExperimentCode.S3].workload.monte_carlo_trials == 120_000
    assert by_code[ExperimentCode.S4].workload.monte_carlo_trials == 50_000
    assert by_code[ExperimentCode.S5].workload.monte_carlo_trials == 120_000
    assert by_code[ExperimentCode.S6].workload.exact_cells == 45
