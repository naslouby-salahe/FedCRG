"""Single-thread benchmark harness for the R13 protocol primitives."""

from __future__ import annotations

import math
import os
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    warmups: int
    repetitions: int
    median_seconds: float
    p95_seconds: float
    peak_bytes: int


def configure_single_thread_execution() -> int | None:
    """Constrain numerical libraries and, when supported, process affinity to one CPU."""

    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        if available:
            cpu = available[0]
            os.sched_setaffinity(0, {cpu})
            return cpu
    return None


def benchmark(
    name: str,
    operation: Callable[[], object],
    warmups: int = 100,
    repetitions: int = 1000,
) -> BenchmarkResult:
    """Measure one primitive after the locked warm-up period."""

    if warmups < 0 or repetitions <= 0:
        raise ValueError("warmups must be nonnegative and repetitions must be positive")
    for _ in range(warmups):
        operation()

    tracemalloc.start()
    timings: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        operation()
        timings.append((time.perf_counter_ns() - started) / 1e9)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ordered = sorted(timings)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return BenchmarkResult(
        name=name,
        warmups=warmups,
        repetitions=repetitions,
        median_seconds=statistics.median(ordered),
        p95_seconds=ordered[p95_index],
        peak_bytes=peak,
    )
