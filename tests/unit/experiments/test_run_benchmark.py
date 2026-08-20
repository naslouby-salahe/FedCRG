from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from fedcrg.config import ExperimentSpec, ParameterAxis, Study
from fedcrg.experiments.analyses import (
    BenchmarkPrimitive,
    BenchmarkReport,
    RunBenchmark,
    TimedRepetitions,
    _timed_repetitions,
)
from fedcrg.types import DatasetId, ExperimentAxisId, ExperimentId, ExperimentType


@pytest.fixture(scope="module")
def base_config():
    return Study.load().resolve(ExperimentId.READINESS_THEOREM)


@pytest.fixture
def benchmark_spec() -> ExperimentSpec:
    return ExperimentSpec(
        id=ExperimentId.COMPUTATIONAL_BENCHMARK,
        category=ExperimentType.BENCHMARK,
        dataset=DatasetId.SYNTHETIC,
        axes=[
            ParameterAxis(id=ExperimentAxisId.WARMUPS, values=[1]),
            ParameterAxis(id=ExperimentAxisId.REPETITIONS, values=[2]),
        ],
    )


def test_run_writes_a_benchmark_report_with_one_cell_per_primitive(
    tmp_path: Path, base_config, benchmark_spec
) -> None:
    output = RunBenchmark(benchmark_spec, base_config).run(tmp_path / "benchmark.json")
    report = BenchmarkReport.model_validate_json(output.read_text())
    assert report.experiment_id is ExperimentId.COMPUTATIONAL_BENCHMARK
    assert {cell.primitive for cell in report.cells} == set(BenchmarkPrimitive)
    assert all(cell.warmups == 1 for cell in report.cells)
    assert all(cell.repetitions == 2 for cell in report.cells)
    assert all(cell.median_seconds >= 0.0 for cell in report.cells)
    assert all(cell.p95_seconds >= cell.median_seconds - 1e-9 for cell in report.cells)


def test_environment_pin_reports_platform_and_library_versions(base_config, benchmark_spec) -> None:
    pin = RunBenchmark(benchmark_spec, base_config)._environment_pin()
    assert "numpy" in pin
    assert "scipy" in pin
    assert "python" in pin


def test_pin_single_cpu_does_not_raise(base_config, benchmark_spec) -> None:
    RunBenchmark._pin_single_cpu()


def test_pin_single_cpu_swallows_affinity_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise OSError("affinity not supported")

    monkeypatch.setattr(os, "sched_setaffinity", _raise, raising=False)
    RunBenchmark._pin_single_cpu()


def test_sample_count_uses_reference_split_for_reference_construction(base_config) -> None:
    split = base_config.dataset.split
    assert (
        RunBenchmark._sample_count(BenchmarkPrimitive.REFERENCE_CONSTRUCTION, split)
        == split.reference_benign
    )


def test_sample_count_uses_calibration_split_for_readiness_and_decision(base_config) -> None:
    split = base_config.dataset.split
    assert (
        RunBenchmark._sample_count(BenchmarkPrimitive.READINESS_ORDER_STATISTIC, split)
        == split.calibration_benign
    )
    assert (
        RunBenchmark._sample_count(BenchmarkPrimitive.DEPLOYMENT_DECISION, split)
        == split.calibration_benign
    )


def test_sample_count_uses_mismatch_split_for_mismatch_evidence(base_config) -> None:
    split = base_config.dataset.split
    assert (
        RunBenchmark._sample_count(BenchmarkPrimitive.MISMATCH_EVIDENCE, split)
        == split.mismatch_benign
    )


def test_sample_count_rejects_an_unrecognized_primitive(base_config) -> None:
    with pytest.raises(AssertionError):
        RunBenchmark._sample_count("not_a_primitive", base_config.dataset.split)


class _DummyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: int


def test_timed_repetitions_reports_median_p95_and_peak_rss() -> None:
    calls = {"count": 0}

    def primitive() -> _DummyResult:
        calls["count"] += 1
        return _DummyResult(value=calls["count"])

    timings = _timed_repetitions(primitive, 5, 95)
    assert isinstance(timings, TimedRepetitions)
    assert calls["count"] == 5
    assert timings.median_seconds >= 0.0
    assert timings.p95_seconds >= 0.0
    assert timings.peak_rss_bytes > 0
