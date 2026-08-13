"""Campaign execution with persistent status and automatic results building.

A campaign is an ordered set of experiment work items. Execution records persistent
status (current experiment, stage, per-experiment outcomes, reuse flags, seeds, elapsed
time, artifact paths). After all required work completes successfully the campaign
invokes the same results builder used by ``fedcrg results build``, so there is exactly
one results-building implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.json_io import atomic_write_json
from fedcrg.configuration.resolve import load_config
from fedcrg.domain.enums import CampaignStatusValue, ExperimentId
from fedcrg.domain.errors import ConfigurationError
from fedcrg.reporting.results import ResultsBuilder
from fedcrg.runtime.logging import get_logger
from fedcrg.runtime.monitoring import ResourceMonitor, write_telemetry

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExperimentCampaignStatus:
    experiment_id: ExperimentId
    status: CampaignStatusValue
    started_at: str | None = None
    finished_at: str | None = None
    problem: str | None = None
    run_directories: tuple[str, ...] = ()
    reused_preprocessing: bool = False
    reused_models: bool = False
    reused_scores: bool = False


@dataclass(frozen=True, slots=True)
class CampaignStatus:
    campaign_id: str
    created_at: str
    updated_at: str
    current_experiment: str | None = None
    current_stage: str | None = None
    experiments: tuple[ExperimentCampaignStatus, ...] = ()
    results_path: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def completed_experiments(self) -> tuple[str, ...]:
        return tuple(
            item.experiment_id.value
            for item in self.experiments
            if item.status is CampaignStatusValue.COMPLETE
        )

    @property
    def pending_experiments(self) -> tuple[str, ...]:
        return tuple(
            item.experiment_id.value
            for item in self.experiments
            if item.status is CampaignStatusValue.PENDING
        )

    @property
    def failed_experiments(self) -> tuple[str, ...]:
        return tuple(
            item.experiment_id.value
            for item in self.experiments
            if item.status is CampaignStatusValue.FAILED
        )

    @property
    def blocked_experiments(self) -> tuple[str, ...]:
        return tuple(
            item.experiment_id.value
            for item in self.experiments
            if item.status is CampaignStatusValue.BLOCKED
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_experiment": self.current_experiment,
            "current_stage": self.current_stage,
            "experiments": [
                {
                    "experiment_id": item.experiment_id.value,
                    "status": item.status.value,
                    "started_at": item.started_at,
                    "finished_at": item.finished_at,
                    "problem": item.problem,
                    "run_directories": list(item.run_directories),
                    "reused_preprocessing": item.reused_preprocessing,
                    "reused_models": item.reused_models,
                    "reused_scores": item.reused_scores,
                }
                for item in self.experiments
            ],
            "results_path": self.results_path,
            "elapsed_seconds": self.elapsed_seconds,
        }


class CampaignStatusStore:
    """Persist campaign status as an atomically written JSON snapshot."""

    def __init__(self, campaigns_root: Path | None = None) -> None:
        self.campaigns_root = campaigns_root or Path("outputs/campaigns")

    def path_for(self, campaign_id: str) -> Path:
        if not campaign_id or "/" in campaign_id or ".." in campaign_id:
            raise ConfigurationError(f"Invalid campaign id: {campaign_id!r}")
        return self.campaigns_root / f"{campaign_id}.json"

    def save(self, status: CampaignStatus) -> Path:
        path = self.path_for(status.campaign_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, status.to_dict())
        return path

    def load(self, campaign_id: str) -> CampaignStatus:
        path = self.path_for(campaign_id)
        if not path.is_file():
            raise ConfigurationError(f"Campaign has no recorded status: {campaign_id}")
        payload = _read_json(path)
        experiments = tuple(
            ExperimentCampaignStatus(
                experiment_id=ExperimentId(str(item["experiment_id"])),
                status=CampaignStatusValue(str(item["status"])),
                started_at=_optional_str(item.get("started_at")),
                finished_at=_optional_str(item.get("finished_at")),
                problem=_optional_str(item.get("problem")),
                run_directories=_string_tuple(item.get("run_directories")),
                reused_preprocessing=bool(item.get("reused_preprocessing", False)),
                reused_models=bool(item.get("reused_models", False)),
                reused_scores=bool(item.get("reused_scores", False)),
            )
            for item in _mapping_list(payload.get("experiments"))
        )
        return CampaignStatus(
            campaign_id=str(payload["campaign_id"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            current_experiment=_optional_str(payload.get("current_experiment")),
            current_stage=_optional_str(payload.get("current_stage")),
            experiments=experiments,
            results_path=_optional_str(payload.get("results_path")),
            elapsed_seconds=_optional_float(payload.get("elapsed_seconds"), 0.0),
        )


def _read_json(path: Path) -> dict[str, object]:
    import json

    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Campaign status must be a JSON object: {path}")
    result: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(key, str):
            result[key] = value
    return result


def _optional_str(value: object | None) -> str | None:
    return str(value) if value is not None else None


def _optional_float(value: object | None, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _mapping_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict):
            rows.append({str(key): entry for key, entry in item.items() if isinstance(key, str)})
    return rows


@dataclass(frozen=True, slots=True)
class CampaignWorkItem:
    experiment_id: ExperimentId
    config_path: Path
    prepared_root: Path | None = None


class CampaignRunner:
    """Execute ordered campaign work items, record status, and build results."""

    def __init__(
        self,
        store: CampaignStatusStore | None = None,
        results_builder: ResultsBuilder | None = None,
    ) -> None:
        self.store = store or CampaignStatusStore()
        self.results_builder = results_builder or ResultsBuilder()
        self._monitor = ResourceMonitor()

    def run(
        self,
        campaign_id: str,
        work_items: tuple[CampaignWorkItem, ...],
        *,
        outputs_root: Path,
        results_root: Path = Path("results"),
    ) -> CampaignStatus:
        if not work_items:
            raise ConfigurationError("A campaign requires at least one work item")
        started = time.monotonic()
        created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        statuses: list[ExperimentCampaignStatus] = [
            ExperimentCampaignStatus(
                experiment_id=item.experiment_id, status=CampaignStatusValue.PENDING
            )
            for item in work_items
        ]
        current_status = CampaignStatus(
            campaign_id=campaign_id,
            created_at=created_at,
            updated_at=created_at,
            current_stage="starting",
            experiments=tuple(statuses),
        )
        self.store.save(current_status)

        failed: list[str] = []
        for index, item in enumerate(work_items):
            if self._dependency_failed(item, failed):
                current_status = self._set_status(
                    current_status,
                    index,
                    status=CampaignStatusValue.BLOCKED,
                    stage=f"blocked {item.experiment_id.value}",
                    problem=f"dependency failed: {', '.join(failed)}",
                    elapsed=time.monotonic() - started,
                )
                self.store.save(current_status)
                continue
            self._record_telemetry(outputs_root)
            running = self._set_status(
                current_status,
                index,
                status=CampaignStatusValue.RUNNING,
                stage=f"running {item.experiment_id.value}",
            )
            current_status = running
            try:
                run_dirs = self._execute_item(item)
            except Exception as exc:  # pragma: no cover - failure path
                _LOGGER.exception("campaign item failed: %s", item.experiment_id.value)
                failed.append(item.experiment_id.value)
                current_status = self._set_status(
                    current_status,
                    index,
                    status=CampaignStatusValue.FAILED,
                    stage=f"failed {item.experiment_id.value}",
                    problem=str(exc),
                    finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    elapsed=time.monotonic() - started,
                )
            else:
                current_status = self._set_status(
                    current_status,
                    index,
                    status=CampaignStatusValue.COMPLETE,
                    stage=f"complete {item.experiment_id.value}",
                    run_directories=tuple(str(path) for path in run_dirs),
                    finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    elapsed=time.monotonic() - started,
                )
            self.store.save(current_status)

        # Mark dependent experiments blocked when a required dependency failed.
        current_status = self._mark_blocked(current_status, work_items, failed)
        if failed:
            self.store.save(current_status)
            raise RuntimeError(f"Campaign {campaign_id} failed: {', '.join(failed)}")

        # All work complete: invoke the exact same builder used by `results build`.
        results_path = self.results_builder.build(
            campaign_id=campaign_id,
            outputs_root=outputs_root,
            results_root=results_root,
        )
        current_status = CampaignStatus(
            campaign_id=current_status.campaign_id,
            created_at=current_status.created_at,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            current_experiment=None,
            current_stage="results built",
            experiments=current_status.experiments,
            results_path=str(results_path),
            elapsed_seconds=time.monotonic() - started,
        )
        self.store.save(current_status)
        return current_status

    @staticmethod
    def _dependency_failed(item: CampaignWorkItem, failed: list[str]) -> bool:
        from fedcrg.experiments.experiment_definition import get_experiment_definition

        dependencies = get_experiment_definition(item.experiment_id).dependencies
        return any(dependency.value in failed for dependency in dependencies)

    def _execute_item(self, item: CampaignWorkItem) -> tuple[Path, ...]:
        """Execute one work item through the single execution spine."""
        from fedcrg.experiments.runner import RunAllExperiments

        config = load_config(item.config_path)
        if config.id is not item.experiment_id:
            raise ConfigurationError(
                f"Work item experiment mismatch: {item.experiment_id.value} vs config {config.id.value}"
            )
        if item.prepared_root is None:
            raise ConfigurationError(
                f"Work item {item.experiment_id.value} requires a prepared root"
            )
        execution = RunAllExperiments().execute(item.experiment_id, config, item.prepared_root)
        return execution.workload.run_directories

    def _set_status(
        self,
        current: CampaignStatus,
        index: int,
        *,
        status: CampaignStatusValue,
        stage: str,
        problem: str | None = None,
        run_directories: tuple[str, ...] = (),
        finished_at: str | None = None,
        elapsed: float | None = None,
    ) -> CampaignStatus:
        rows = list(current.experiments)
        item = rows[index]
        rows[index] = ExperimentCampaignStatus(
            experiment_id=item.experiment_id,
            status=status,
            started_at=item.started_at or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            finished_at=finished_at,
            problem=problem,
            run_directories=run_directories or item.run_directories,
            reused_preprocessing=item.reused_preprocessing,
            reused_models=item.reused_models,
            reused_scores=item.reused_scores,
        )
        return CampaignStatus(
            campaign_id=current.campaign_id,
            created_at=current.created_at,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            current_experiment=item.experiment_id.value,
            current_stage=stage,
            experiments=tuple(rows),
            results_path=current.results_path,
            elapsed_seconds=elapsed if elapsed is not None else current.elapsed_seconds,
        )

    def _mark_blocked(
        self,
        current: CampaignStatus,
        work_items: tuple[CampaignWorkItem, ...],
        failed: list[str],
    ) -> CampaignStatus:
        if not failed:
            return current
        rows = list(current.experiments)
        for index, item in enumerate(work_items):
            if rows[index].status is CampaignStatusValue.PENDING:
                rows[index] = ExperimentCampaignStatus(
                    experiment_id=item.experiment_id,
                    status=CampaignStatusValue.BLOCKED,
                    problem=f"dependency failed: {', '.join(failed)}",
                )
        return CampaignStatus(
            campaign_id=current.campaign_id,
            created_at=current.created_at,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            current_experiment=current.current_experiment,
            current_stage="blocked by failed dependencies",
            experiments=tuple(rows),
            results_path=current.results_path,
            elapsed_seconds=current.elapsed_seconds,
        )

    def _record_telemetry(self, outputs_root: Path) -> None:
        sample = self._monitor.sample()
        write_telemetry(sample, outputs_root / "monitoring" / "telemetry.jsonl")
