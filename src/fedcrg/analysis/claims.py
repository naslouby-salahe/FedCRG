"""Pre-registered claim-strength gates G1-G8 and claim-level classification."""

from __future__ import annotations

from dataclasses import dataclass

from fedcrg.core.enums import ClaimLevel


@dataclass(frozen=True, slots=True)
class ClaimGateEvidence:
    statistical_core_integrity: bool
    data_integrity: bool
    reliability_benefit: bool
    two_component_incremental_value: bool
    external_replication: bool
    detector_robustness: bool
    assumption_stresses_reported: bool
    reproducibility: bool


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    level: ClaimLevel
    failed_gates: tuple[str, ...]


def assess_claim_level(evidence: ClaimGateEvidence) -> ClaimAssessment:
    gates = {
        "G1": evidence.statistical_core_integrity,
        "G2": evidence.data_integrity,
        "G3": evidence.reliability_benefit,
        "G4": evidence.two_component_incremental_value,
        "G5": evidence.external_replication,
        "G6": evidence.detector_robustness,
        "G7": evidence.assumption_stresses_reported,
        "G8": evidence.reproducibility,
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    if not gates["G1"] or not gates["G2"] or not gates["G8"]:
        level = ClaimLevel.INVALID
    elif all(gates.values()):
        level = ClaimLevel.METHOD_BENEFIT
    elif all(gates[name] for name in ("G1", "G2", "G3", "G4", "G7", "G8")):
        level = ClaimLevel.DATASET_LIMITED_BENEFIT
    else:
        level = ClaimLevel.CHARACTERIZATION
    return ClaimAssessment(level, failed)
