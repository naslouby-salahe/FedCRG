"""Pre-registered claim-strength gates G0-G8, claim-level classification, and their
derivation from frozen experiment evidence."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fedcrg.analysis.policy_contrasts import FederationResultRecord, load_federation_results
from fedcrg.artifacts.paths import RunLayout
from fedcrg.artifacts.manifests import RunManifestStore
from fedcrg.artifacts.json_io import atomic_write_json
from fedcrg.artifacts.integrity import ArtifactVerifier
from fedcrg.configuration.experiment_config import ExperimentConfig
from fedcrg.configuration.resolve import ExperimentConfigResolver
from fedcrg.configuration.statistics_config import StatisticsConfig
from fedcrg.domain.enums import (
    ClaimLevel,
    ExperimentId,
    ExperimentStatus,
    PolicyId,
)
from fedcrg.evaluation.federation_evaluation import utility_margin_satisfied
from fedcrg.experiments.completion import ExperimentCompletion, ExperimentCompletionAuditor
from fedcrg.experiments.verification import VerifyOutputs


@dataclass(frozen=True, slots=True)
class ClaimGateEvidence:
    """Evidence flags corresponding exactly to the pre-registered claim-strength gates."""

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
        () if gates["G0"] else ("G0: submission-week novelty recheck is not current",)
    )
    return ClaimAssessment(level, failed, release_blockers)


@dataclass(frozen=True, slots=True)
class GateDiagnostic:
    gate: str
    message: str


@dataclass(frozen=True, slots=True)
class ClaimGateReport:
    evidence: ClaimGateEvidence
    assessment: ClaimAssessment
    diagnostics: tuple[GateDiagnostic, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence": asdict(self.evidence),
            "assessment": {
                "level": self.assessment.level.value,
                "failed_gates": list(self.assessment.failed_gates),
                "release_blockers": list(self.assessment.release_blockers),
            },
            "diagnostics": {item.gate: item.message for item in self.diagnostics},
        }


class ClaimGateEvaluator:
    """Apply G0-G8 without outcome-conditioned redesign or hidden success criteria."""

    def __init__(
        self,
        completion: ExperimentCompletionAuditor | None = None,
        verifier: ArtifactVerifier | None = None,
    ) -> None:
        self.completion = completion or ExperimentCompletionAuditor()
        self.verifier = verifier or ArtifactVerifier()
        self.manifests = RunManifestStore()

    def evaluate(
        self,
        outputs_root: Path,
        *,
        novelty_log: Path | None = None,
        repository_certified: bool = False,
        repository_root: Path = Path("."),
        statistics: StatisticsConfig | None = None,
        named_calibration_seeds: dict[ExperimentId, int] | None = None,
    ) -> ClaimGateReport:
        """Evaluate the claim gates against frozen evidence.

        Statistical choices (utility margin, familywise alpha) come from the resolved
        statistics configuration. Named calibration splits come from the frozen dataset
        profiles.
        """
        primary_config = self._load_config(
            repository_root, "configs/experiments/primary/nbaiot.yaml"
        )
        resolved_statistics = statistics or primary_config.statistics
        named_seeds = named_calibration_seeds or {
            ExperimentId.PRIMARY_NBAIOT: primary_config.dataset.primary_calibration_seed,
            ExperimentId.EXTERNAL_DIAD: self._load_config(
                repository_root, "configs/experiments/external/diad.yaml"
            ).dataset.primary_calibration_seed,
        }
        completion = {
            row.experiment_id: row for row in self.completion.audit(outputs_root, repository_root)
        }
        run_dirs = tuple(self._completed_run_dirs(outputs_root / "runs"))
        records = load_federation_results(run_dirs)
        diagnostics: dict[str, str] = {}

        g0 = self._novelty_recheck_current(novelty_log)
        diagnostics["G0"] = (
            "submission-week novelty log is current"
            if g0
            else "no current <=7-day novelty-recheck evidence was supplied"
        )

        g1 = self._statistical_core_integrity(outputs_root, completion)
        diagnostics["G1"] = (
            "exact precompute and S1 evidence reconcile"
            if g1
            else "statistical-core evidence is incomplete or S1 contains a rejected cell"
        )

        g2 = self._data_integrity(completion, run_dirs)
        diagnostics["G2"] = (
            "R1 workload and run provenance verify"
            if g2
            else "primary data/run integrity evidence is incomplete"
        )

        g3 = self._reliability_claim(
            records,
            ExperimentId.PRIMARY_NBAIOT,
            named_seeds[ExperimentId.PRIMARY_NBAIOT],
            resolved_statistics,
        )
        diagnostics["G3"] = (
            "primary reliability benefit with locked utility margin is supported"
            if g3
            else "primary reliability/utility gate is not supported"
        )

        g4 = self._two_component_value(records, run_dirs, named_seeds)
        diagnostics["G4"] = (
            "mismatch evidence changes at least one decision and improves the readiness-only ablation"
            if g4
            else "two-component incremental-value gate is not supported"
        )

        g5 = self._external_replication(records, completion, resolved_statistics)
        diagnostics["G5"] = (
            "external reliability direction and utility margin replicate"
            if g5
            else "external replication gate is incomplete or unsupported"
        )

        g6 = self._detector_robustness(records, completion)
        diagnostics["G6"] = (
            "second-detector reliability direction is qualitatively consistent"
            if g6
            else "second-detector robustness gate is incomplete or unsupported"
        )

        required_stress = (
            ExperimentId.TEMPORAL_DEPENDENCE,
            ExperimentId.CALIBRATION_SHIFT,
            ExperimentId.CALIBRATION_CONTAMINATION,
            ExperimentId.SOURCE_ORDER_TEST,
            ExperimentId.REAL_CONTAMINATION,
            ExperimentId.SOURCE_ORDER_CALIBRATION,
        )
        g7 = all(
            completion.get(code) is not None and completion[code].complete
            for code in required_stress
        )
        diagnostics["G7"] = (
            "all locked assumption stresses are present"
            if g7
            else "one or more locked assumption stresses are missing"
        )

        g8 = repository_certified
        diagnostics["G8"] = (
            "clean-checkout reproducibility certification passed"
            if g8
            else "full clean-checkout certification has not been supplied"
        )

        evidence = ClaimGateEvidence(
            novelty_recheck_current=g0,
            statistical_core_integrity=g1,
            data_integrity=g2,
            reliability_benefit=g3,
            two_component_incremental_value=g4,
            external_replication=g5,
            detector_robustness=g6,
            assumption_stresses_reported=g7,
            reproducibility=g8,
        )
        return ClaimGateReport(
            evidence,
            assess_claim_level(evidence),
            tuple(GateDiagnostic(gate, message) for gate, message in diagnostics.items()),
        )

    def write(
        self,
        outputs_root: Path,
        output: Path,
        *,
        novelty_log: Path | None = None,
        repository_certified: bool = False,
    ) -> Path:
        report = self.evaluate(
            outputs_root,
            novelty_log=novelty_log,
            repository_certified=repository_certified,
        )
        atomic_write_json(output, report.to_dict())
        return output

    @staticmethod
    def _novelty_recheck_current(path: Path | None) -> bool:
        if path is None or not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            performed = datetime.fromisoformat(str(payload["performed_at"]).replace("Z", "+00:00"))
            if performed.tzinfo is None:
                performed = performed.replace(tzinfo=UTC)
            age_days = (datetime.now(UTC) - performed.astimezone(UTC)).total_seconds() / 86400
            queries = payload.get("queries")
            sources = payload.get("sources_checked")
            closer = bool(payload.get("closer_method_found", False))
            wording_updated = bool(payload.get("novelty_wording_updated", False))
            return (
                0.0 <= age_days <= 7.0
                and isinstance(queries, list)
                and bool(queries)
                and isinstance(sources, list)
                and bool(sources)
                and (not closer or wording_updated)
            )
        except (KeyError, TypeError, ValueError):
            return False

    def _statistical_core_integrity(
        self,
        outputs_root: Path,
        completion: dict[ExperimentId, ExperimentCompletion],
    ) -> bool:
        precompute = VerifyOutputs.verify_protocol_precompute(outputs_root)
        s1 = completion.get(ExperimentId.READINESS_THEOREM)
        if not precompute.valid or s1 is None or not s1.complete:
            return False
        path = outputs_root / "experiments" / ExperimentId.READINESS_THEOREM.value / "results.json"
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells = payload.get("cells")
        return (
            isinstance(cells, list)
            and bool(cells)
            and all(isinstance(cell, dict) and cell.get("accepted") is True for cell in cells)
        )

    def _data_integrity(
        self,
        completion: dict[ExperimentId, ExperimentCompletion],
        run_dirs: tuple[Path, ...],
    ) -> bool:
        r1 = completion.get(ExperimentId.PRIMARY_NBAIOT)
        if r1 is None or not r1.complete:
            return False
        primary = [
            path
            for path in run_dirs
            if self.manifests.load(RunLayout(path).manifest).experiment_id
            is ExperimentId.PRIMARY_NBAIOT
        ]
        return bool(primary) and all(
            self.verifier.verify(RunLayout(path)).valid for path in primary
        )

    @staticmethod
    def _cell_records(
        records: tuple[FederationResultRecord, ...],
        experiment: ExperimentId,
        calibration_seed: int,
    ) -> tuple[FederationResultRecord, ...]:
        return tuple(
            row
            for row in records
            if row.experiment_id == experiment.value and row.calibration_seed == calibration_seed
        )

    @staticmethod
    def _load_config(repository_root: Path, relative: str) -> ExperimentConfig:
        return ExperimentConfigResolver().resolve(repository_root / relative)

    @classmethod
    def _reliability_claim(
        cls,
        records: tuple[FederationResultRecord, ...],
        experiment: ExperimentId,
        calibration_seed: int,
        statistics: StatisticsConfig,
    ) -> bool:
        selected = cls._cell_records(records, experiment, calibration_seed)
        seeds = {row.model_seed for row in selected if row.policy is PolicyId.FEDCRG}
        if not seeds:
            return False
        method_mebe = [row.mebe for row in selected if row.policy is PolicyId.FEDCRG]
        global_mebe = [row.mebe for row in selected if row.policy is PolicyId.GLOBAL_QUANTILE]
        local_mebe = [row.mebe for row in selected if row.policy is PolicyId.LOCAL_QUANTILE]
        if not method_mebe or not global_mebe or not local_mebe:
            return False
        method_mean = float(np.mean(method_mebe))
        reliability_better = method_mean < float(np.mean(global_mebe)) or method_mean < float(
            np.mean(local_mebe)
        )
        for seed in seeds:
            rows = [row for row in selected if row.model_seed == seed]
            method = next(
                (row.attack_balanced_macro_tpr for row in rows if row.policy is PolicyId.FEDCRG),
                None,
            )
            anchors = [
                row.attack_balanced_macro_tpr
                for row in rows
                if row.policy
                in {
                    PolicyId.GLOBAL_QUANTILE,
                    PolicyId.LOCAL_QUANTILE,
                    PolicyId.SHRINKAGE,
                }
                and row.attack_balanced_macro_tpr is not None
            ]
            if (
                method is None
                or not anchors
                or not utility_margin_satisfied(
                    method, max(anchors), statistics.utility_margin_allowance
                )
            ):
                return False
        return bool(reliability_better)

    def _two_component_value(
        self,
        records: tuple[FederationResultRecord, ...],
        run_dirs: tuple[Path, ...],
        named_seeds: dict[ExperimentId, int],
    ) -> bool:
        natural = (
            (ExperimentId.PRIMARY_NBAIOT, named_seeds[ExperimentId.PRIMARY_NBAIOT]),
            (ExperimentId.EXTERNAL_DIAD, named_seeds[ExperimentId.EXTERNAL_DIAD]),
        )
        for experiment, calibration_seed in natural:
            selected = self._cell_records(records, experiment, calibration_seed)
            method = [row for row in selected if row.policy is PolicyId.FEDCRG]
            readiness = [row for row in selected if row.policy is PolicyId.READINESS_ONLY]
            if not method or not readiness:
                continue
            method_mebe = float(np.mean([row.mebe for row in method]))
            readiness_mebe = float(np.mean([row.mebe for row in readiness]))
            method_bvr = float(np.mean([row.band_violation_rate for row in method]))
            readiness_bvr = float(np.mean([row.band_violation_rate for row in readiness]))
            if not (method_mebe < readiness_mebe or method_bvr < readiness_bvr):
                continue
            if self._policy_thresholds_differ(run_dirs, experiment, calibration_seed):
                return True
        return False

    def _policy_thresholds_differ(
        self,
        run_dirs: tuple[Path, ...],
        experiment: ExperimentId,
        calibration_seed: int,
    ) -> bool:
        by_key: dict[tuple[int, str], dict[PolicyId, float | None]] = {}
        for path in run_dirs:
            manifest = self.manifests.load(RunLayout(path).manifest)
            if (
                manifest.experiment_id is not experiment
                or manifest.calibration_seed != calibration_seed
                or manifest.policy_id not in {PolicyId.FEDCRG, PolicyId.READINESS_ONLY}
            ):
                continue
            for line in RunLayout(path).threshold_records.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                row = json.loads(line)
                key = (manifest.model_seed, str(row["client_id"]))
                by_key.setdefault(key, {})[manifest.policy_id] = row.get("selected_tau")
        return any(
            PolicyId.FEDCRG in values
            and PolicyId.READINESS_ONLY in values
            and values[PolicyId.FEDCRG] != values[PolicyId.READINESS_ONLY]
            for values in by_key.values()
        )

    @classmethod
    def _external_replication(
        cls,
        records: tuple[FederationResultRecord, ...],
        completion: dict[ExperimentId, ExperimentCompletion],
        statistics: StatisticsConfig,
    ) -> bool:
        r10 = completion.get(ExperimentId.EXTERNAL_DIAD)
        return bool(
            r10 is not None
            and r10.complete
            and cls._reliability_claim(records, ExperimentId.EXTERNAL_DIAD, 2000, statistics)
        )

    @classmethod
    def _detector_robustness(
        cls,
        records: tuple[FederationResultRecord, ...],
        completion: dict[ExperimentId, ExperimentCompletion],
    ) -> bool:
        r11 = completion.get(ExperimentId.SECOND_DETECTOR)
        if r11 is None or not r11.complete:
            return False
        selected = cls._cell_records(records, ExperimentId.SECOND_DETECTOR, 1000)
        method = [row.mebe for row in selected if row.policy is PolicyId.FEDCRG]
        anchors = {
            policy: [row.mebe for row in selected if row.policy is policy]
            for policy in (
                PolicyId.GLOBAL_QUANTILE,
                PolicyId.LOCAL_QUANTILE,
                PolicyId.SHRINKAGE,
            )
        }
        return bool(
            method
            and any(values and np.mean(method) < np.mean(values) for values in anchors.values())
        )

    def _completed_run_dirs(self, root: Path) -> Iterator[Path]:
        if not root.exists():
            return
        for path in sorted(item for item in root.iterdir() if item.is_dir()):
            manifest_path = RunLayout(path).manifest
            if not manifest_path.is_file():
                continue
            manifest = self.manifests.load(manifest_path)
            if manifest.status is ExperimentStatus.COMPLETE:
                yield path
