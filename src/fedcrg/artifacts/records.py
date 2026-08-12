"""Normalized threshold and metric evidence records required by the protocol."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fedcrg.core.enums import PolicyId


@dataclass(frozen=True, slots=True)
class ThresholdRecord:
    run_id: str
    policy_id: PolicyId
    client_id: str
    tau_ref: float
    tau_local: float | None
    selected_tau: float | None
    readiness_n: int
    readiness_rank: int
    readiness_probability: float
    mismatch_n: int
    mismatch_x: int
    cp_lower: float
    cp_upper: float
    p_low: float | None
    p_high: float
    state: str
    tie_count: int
    selected_source: str
    reason_code: str


@dataclass(frozen=True, slots=True)
class MetricRecord:
    run_id: str
    policy_id: PolicyId
    client_id: str
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
            payload = asdict(record)
            payload["policy_id"] = record.policy_id.value
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    temp.replace(path)
