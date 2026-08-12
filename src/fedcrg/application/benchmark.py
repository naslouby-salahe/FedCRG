"""R13 benchmark application using precomputed and cached protocol primitives."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fedcrg.analysis.benchmark import benchmark
from fedcrg.artifacts.serialization import atomic_write_json


class RunBenchmark:
    def run(self, operations: dict[str, Callable[[], object]], output: Path) -> Path:
        rows = [asdict(benchmark(name, operation, 100, 1000)) for name, operation in operations.items()]
        atomic_write_json(output, {"warmups": 100, "repetitions": 1000, "results": rows})
        return output
