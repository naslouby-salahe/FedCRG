"""Single-thread benchmark harness for R13 primitives."""

from __future__ import annotations

import os
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    warmups: int
    repetitions: int
    median_seconds: float
    p95_seconds: float
    peak_bytes: int


def benchmark(name: str, operation: Callable[[], object], warmups: int = 100, repetitions: int = 1000) -> BenchmarkResult:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
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
    p95_index = min(len(ordered) - 1, int(np_ceil(0.95 * len(ordered))) - 1)
    return BenchmarkResult(name, warmups, repetitions, statistics.median(ordered), ordered[p95_index], peak)


def np_ceil(value: float) -> int:
    return int(value) if value == int(value) else int(value) + 1
