"""Normalized threshold/metric evidence records, cache references, and experiment-level
evidence envelopes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from fedcrg.artifacts.integrity import sha256_file
from fedcrg.artifacts.json_io import JsonObject, JsonValue, atomic_write_json, to_json_value
from fedcrg.domain.enums import (
    DecisionReason,
    DecisionState,
    ExperimentId,
    PolicyId,
    ThresholdSource,
)
from fedcrg.domain.identifiers import ClientId, RunId, Sha256


@dataclass(frozen=True, slots=True)
class ThresholdRecord:
    run_id: RunId
    policy_id: PolicyId
    client_id: ClientId
    tau_ref: float
    tau_local: float | None
    selected_tau: float | None
    readiness_n: int
    readiness_rank: int
    readiness_probability: float
    mismatch_n: int
    mismatch_x: int
    cp_lower: float | None
    cp_upper: float | None
    p_low: float | None
    p_high: float
    state: DecisionState
    tie_count: int
    selected_source: ThresholdSource
    reason_code: DecisionReason


@dataclass(frozen=True, slots=True)
class MetricRecord:
    run_id: RunId
    policy_id: PolicyId
    client_id: ClientId
    benign_n: int
    attack_n: int
    fp: int
    tn: int
    tp: int
    fn: int
    fpr: float
    tpr: float | None
    precision: float | None
    f1: float | None
    balanced_accuracy: float | None
    auroc: float
    auprc: float
    band_error: float
    attack_balanced_tpr: float | None


def write_jsonl(path: Path, records: tuple[ThresholdRecord | MetricRecord, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_json_value(record), sort_keys=True) + "\n")
    temp.replace(path)


@dataclass(frozen=True, slots=True)
class CacheReference:
    """Reference one immutable cache file without duplicating it per policy cell."""

    relative_path: PurePosixPath
    sha256: Sha256

    @classmethod
    def build(cls, file_path: Path, outputs_root: Path) -> CacheReference:
        resolved_file = file_path.resolve()
        resolved_outputs = outputs_root.resolve()
        try:
            relative = resolved_file.relative_to(resolved_outputs)
        except ValueError as exc:
            raise ValueError(
                f"Cache artifact must live under outputs root: {resolved_file}"
            ) from exc
        return cls(PurePosixPath(relative.as_posix()), Sha256(sha256_file(resolved_file)))

    def resolve(self, outputs_root: Path) -> Path:
        path = (outputs_root / Path(self.relative_path)).resolve()
        try:
            path.relative_to(outputs_root.resolve())
        except ValueError as exc:
            raise ValueError("Cache reference escapes the outputs root") from exc
        return path

    def verify(self, outputs_root: Path) -> bool:
        path = self.resolve(outputs_root)
        return path.is_file() and Sha256(sha256_file(path)) == self.sha256


class CacheReferenceStore:
    def save(self, path: Path, reference: CacheReference) -> None:
        atomic_write_json(path, reference)

    def load(self, path: Path) -> CacheReference:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CacheReference(
            relative_path=PurePosixPath(str(payload["relative_path"])),
            sha256=Sha256(str(payload["sha256"])),
        )


class ExperimentResultEnvelope(BaseModel):
    """Self-describing aggregate evidence for one pre-registered S/R experiment.

    Pydantic owns parsing and serialization. ``cells`` hold experiment-specific
    JSON evidence payloads at the artifact boundary.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: ExperimentId
    config_hash: Sha256
    master_seed: int | None
    expected_cells: int | None
    expected_monte_carlo_trials: int
    expected_exact_cells: int
    cells: tuple[JsonObject, ...] = ()
    notes: tuple[str, ...] = ()
    metadata: JsonObject | None = None
    schema_version: int = Field(default=1, frozen=True)

    @property
    def observed_cells(self) -> int:
        return len(self.cells)

    @property
    def observed_monte_carlo_trials(self) -> int:
        total = 0
        for cell in self.cells:
            repetitions = cell.get("repetitions")
            if isinstance(repetitions, int):
                total += repetitions
        return total

    @property
    def complete(self) -> bool:
        if self.expected_cells is not None and self.observed_cells != self.expected_cells:
            return False
        if (
            self.expected_monte_carlo_trials
            and self.observed_monte_carlo_trials != self.expected_monte_carlo_trials
        ):
            return False
        if self.expected_exact_cells and self.observed_cells != self.expected_exact_cells:
            return False
        return True

    def to_dict(self) -> dict[str, JsonValue]:
        payload = self.model_dump(mode="json")
        payload["observed_cells"] = self.observed_cells
        payload["observed_monte_carlo_trials"] = self.observed_monte_carlo_trials
        payload["complete"] = self.complete
        payload["created_at"] = datetime.now(UTC).isoformat()
        return payload

    def write(self, path: Path) -> Path:
        atomic_write_json(path, self.to_dict())
        return path
