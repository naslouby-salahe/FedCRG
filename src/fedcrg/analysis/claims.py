"""Pre-registered claim-strength gates G0-G8 and claim-level classification."""

from __future__ import annotations

from dataclasses import dataclass

from fedcrg.core.enums import ClaimLevel


@dataclass(frozen=True, slots=True)
class ClaimGateEvidence:
    """Evidence flags corresponding exactly to the roadmap claim-strength gates."""

    novelty_recheck_current: bool
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
    release_blockers: tuple[str, ...]


def assess_claim_level(evidence: ClaimGateEvidence) -> ClaimAssessment:
    """Classify the strongest defensible claim without suppressing negative results.

    G0 is a submission-week novelty discipline rather than a scientific outcome gate.
    Its failure blocks release of the novelty wording but does not turn otherwise valid
    experimental evidence into an invalid study. G1, G2, and G8 are the integrity
    gates that make the experimental package invalid when they fail.
    """

    gates = {
        "G0": evidence.novelty_recheck_current,
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
    integrity_gates = ("G1", "G2", "G8")
    scientific_gates = tuple(name for name in gates if name != "G0")

    if any(not gates[name] for name in integrity_gates):
        level = ClaimLevel.INVALID
    elif all(gates[name] for name in scientific_gates):
        level = ClaimLevel.METHOD_BENEFIT
    elif all(gates[name] for name in ("G1", "G2", "G3", "G4", "G7", "G8")):
        level = ClaimLevel.DATASET_LIMITED_BENEFIT
    else:
        level = ClaimLevel.CHARACTERIZATION

    release_blockers = (
        ()
        if gates["G0"]
        else ("G0: submission-week novelty recheck is not current",)
    )
    return ClaimAssessment(level, failed, release_blockers)
