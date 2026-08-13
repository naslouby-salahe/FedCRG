"""Normalized threshold and metric evidence records required by the protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.serialization import to_json_value
from fedcrg.domain.enums import DecisionReason, DecisionState, PolicyId, ThresholdSource
from fedcrg.domain.identifiers import ClientId, RunId


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
