"""Closed-form workload expectations used to reconcile generated artifacts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkloadLedger:
    detector_trainings: int
    policy_cells: int
    method_decisions: int
    reference_constructions: int


def nbaiot_primary_ledger(*, full_split_sensitivity: bool = True) -> WorkloadLedger:
    calibration_count = 50 if full_split_sensitivity else 1
    return WorkloadLedger(
        detector_trainings=5,
        policy_cells=5 * calibration_count * 12 * 9,
        method_decisions=5 * calibration_count * 9,
        reference_constructions=5 * calibration_count,
    )


def deep_svdd_ledger() -> WorkloadLedger:
    return WorkloadLedger(
        detector_trainings=3,
        policy_cells=3 * 10 * 5 * 9,
        method_decisions=3 * 10 * 9,
        reference_constructions=3 * 10,
    )


def diad_primary_ledger(eligible_clients: int) -> WorkloadLedger:
    if eligible_clients <= 0:
        raise ValueError("eligible_clients must be positive")
    return WorkloadLedger(
        detector_trainings=5,
        policy_cells=5 * 20 * 12 * eligible_clients,
        method_decisions=5 * 20 * eligible_clients,
        reference_constructions=5 * 20,
    )
