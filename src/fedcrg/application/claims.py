"""Derive the pre-registered claim-strength gates from frozen experiment evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path

import numpy as np

from fedcrg.analysis.claims import ClaimAssessment, ClaimGateEvidence, assess_claim_level
from fedcrg.analysis.primary import FederationResultRecord, load_federation_results
from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.manifest import RunManifestStore
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.artifacts.verification import ArtifactVerifier
from fedcrg.core.enums import ExperimentCode, ExperimentId, ExperimentStatus, PolicyId
from fedcrg.experiments.completion import ExperimentCompletion, ExperimentCompletionAuditor
from fedcrg.metrics.federation import utility_margin_satisfied


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

    def __init__(self) -> None:
        self.completion = ExperimentCompletionAuditor()
        self.verifier = ArtifactVerifier()
        self.manifests = RunManifestStore()

    def evaluate(
        self,
        outputs_root: Path,
        *,
        novelty_log: Path | None = None,
        repository_certified: bool = False,
    ) -> ClaimGateReport:
        completion = {row.protocol_code: row for row in self.completion.audit(outputs_root)}
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

        g3 = self._reliability_claim(records, ExperimentId.PRIMARY_NBAIOT, 1000)
        diagnostics["G3"] = (
            "primary reliability benefit with locked utility margin is supported"
            if g3
            else "primary reliability/utility gate is not supported"
        )

        g4 = self._two_component_value(records, run_dirs)
        diagnostics["G4"] = (
            "mismatch evidence changes at least one decision and improves the readiness-only ablation"
            if g4
            else "two-component incremental-value gate is not supported"
        )

        g5 = self._external_replication(records, completion)
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
            ExperimentCode.S3,
            ExperimentCode.S4,
            ExperimentCode.S5,
            ExperimentCode.R8,
            ExperimentCode.R9,
            ExperimentCode.R12,
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
        completion: dict[ExperimentCode, ExperimentCompletion],
    ) -> bool:
        from fedcrg.application.verify import VerifyOutputs

        precompute = VerifyOutputs.verify_protocol_precompute(outputs_root)
        s1 = completion.get(ExperimentCode.S1)
        if not precompute.valid or s1 is None or not s1.complete:
            return False
        path = outputs_root / "experiments" / ExperimentCode.S1 / "results.json"
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
        completion: dict[ExperimentCode, ExperimentCompletion],
        run_dirs: tuple[Path, ...],
    ) -> bool:
        r1 = completion.get(ExperimentCode.R1)
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

    @classmethod
    def _reliability_claim(
        cls,
        records: tuple[FederationResultRecord, ...],
        experiment: ExperimentId,
        calibration_seed: int,
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
            if method is None or not anchors or not utility_margin_satisfied(method, max(anchors)):
                return False
        return bool(reliability_better)

    def _two_component_value(
        self,
        records: tuple[FederationResultRecord, ...],
        run_dirs: tuple[Path, ...],
    ) -> bool:
        natural = (
            (ExperimentId.PRIMARY_NBAIOT, 1000),
            (ExperimentId.EXTERNAL_DIAD, 2000),
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
        completion: dict[ExperimentCode, ExperimentCompletion],
    ) -> bool:
        r10 = completion.get(ExperimentCode.R10)
        return bool(
            r10 is not None
            and r10.complete
            and cls._reliability_claim(records, ExperimentId.EXTERNAL_DIAD, 2000)
        )

    @classmethod
    def _detector_robustness(
        cls,
        records: tuple[FederationResultRecord, ...],
        completion: dict[ExperimentCode, ExperimentCompletion],
    ) -> bool:
        r11 = completion.get(ExperimentCode.R11)
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

    def _completed_run_dirs(self, root: Path):
        if not root.exists():
            return
        for path in sorted(item for item in root.iterdir() if item.is_dir()):
            manifest_path = RunLayout(path).manifest
            if not manifest_path.is_file():
                continue
            manifest = self.manifests.load(manifest_path)
            if manifest.status is ExperimentStatus.COMPLETE:
                yield path
