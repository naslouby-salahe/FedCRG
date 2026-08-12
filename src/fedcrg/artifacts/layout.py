"""Structured output paths for one immutable run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fedcrg.core.ids import RunId


@dataclass(frozen=True, slots=True)
class RunLayout:
    root: Path

    @classmethod
    def for_run(cls, outputs_root: Path, run_id: RunId) -> "RunLayout":
        return cls(outputs_root / "runs" / str(run_id))

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def resolved_config(self) -> Path:
        return self.root / "resolved_config.yaml"

    @property
    def environment(self) -> Path:
        return self.root / "environment.json"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def training(self) -> Path:
        return self.root / "training"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def decisions(self) -> Path:
        return self.root / "decisions"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def verification(self) -> Path:
        return self.root / "verification"

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        for directory in (
            self.data,
            self.training,
            self.scores,
            self.decisions,
            self.metrics,
            self.tables,
            self.figures,
            self.reports,
            self.logs,
            self.verification,
        ):
            directory.mkdir()
