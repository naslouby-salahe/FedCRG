"""Canonical immutable output layout for one policy/run cell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fedcrg.domain.identifiers import RunId


@dataclass(frozen=True, slots=True)
class RunLayout:
    root: Path

    @classmethod
    def for_run(cls, outputs_root: Path, run_id: RunId) -> RunLayout:
        return cls(outputs_root / "runs" / str(run_id))

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def run_config(self) -> Path:
        return self.root / "run_config.json"

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
    def model_reference(self) -> Path:
        return self.training / "model_reference.json"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def score_reference(self) -> Path:
        return self.scores / "cache_reference.json"

    @property
    def decisions(self) -> Path:
        return self.root / "decisions"

    @property
    def threshold_records(self) -> Path:
        return self.decisions / "threshold_record.jsonl"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def metric_records(self) -> Path:
        return self.metrics / "metric_record.jsonl"

    @property
    def federation_metrics(self) -> Path:
        return self.metrics / "federation.json"

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
