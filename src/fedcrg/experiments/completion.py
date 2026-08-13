"""Roadmap workload reconciliation for generated experiment evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.manifest import RunManifest, RunManifestStore
from fedcrg.core.enums import ExperimentId, ExperimentStatus, PolicyId
from fedcrg.experiments.definitions import SECOND_DETECTOR_POLICIES
from fedcrg.experiments.registry import ExperimentRegistry

_PRIMARY_MODEL_SEEDS = (11, 22, 33, 44, 55)
_NBAIOT_CALIBRATION_SEEDS = tuple(range(1000, 1050))
_DIAD_CALIBRATION_SEEDS = tuple(range(2000, 2020))
_SECOND_DETECTOR_MODEL_SEEDS = (11, 22, 33)
_SECOND_DETECTOR_CALIBRATION_SEEDS = tuple(range(1000, 1010))


@dataclass(frozen=True, slots=True)
class ExperimentCompletion:
    protocol_code: str
    complete: bool
    expected_cells: int | None
    observed_cells: int
    problems: tuple[str, ...]


class ExperimentCompletionAuditor:
    """Reconcile every pre-registered workload without inferring missing evidence."""

    def __init__(self) -> None:
        self.registry = ExperimentRegistry()
        self.manifests = RunManifestStore()

    def audit(self, outputs_root: Path) -> tuple[ExperimentCompletion, ...]:
        runs = self._completed_runs(outputs_root / "runs")
        rows: list[ExperimentCompletion] = []
        for definition in self.registry.all():
            experiment_id = definition.id
            code = definition.protocol_code
            if experiment_id is ExperimentId.PRIMARY_NBAIOT:
                rows.append(
                    self._policy_run_workload(
                        code,
                        experiment_id,
                        runs,
                        _PRIMARY_MODEL_SEEDS,
                        _NBAIOT_CALIBRATION_SEEDS,
                        tuple(PolicyId),
                        expected_clients=9,
                    )
                )
            elif experiment_id is ExperimentId.EXTERNAL_DIAD:
                rows.append(
                    self._external_policy_workload(
                        code,
                        experiment_id,
                        runs,
                        _PRIMARY_MODEL_SEEDS,
                        _DIAD_CALIBRATION_SEEDS,
                        tuple(PolicyId),
                    )
                )
            elif experiment_id is ExperimentId.SECOND_DETECTOR:
                rows.append(
                    self._policy_run_workload(
                        code,
                        experiment_id,
                        runs,
                        _SECOND_DETECTOR_MODEL_SEEDS,
                        _SECOND_DETECTOR_CALIBRATION_SEEDS,
                        SECOND_DETECTOR_POLICIES,
                        expected_clients=9,
                    )
                )
            elif experiment_id is ExperimentId.DIAD_FEATURE_SENSITIVITY:
                rows.append(
                    self._external_policy_workload(
                        code,
                        experiment_id,
                        runs,
                        _PRIMARY_MODEL_SEEDS,
                        (2000,),
                        definition.policies,
                    )
                )
            elif code in {"S1", "S2", "S3", "S4", "S5", "S6", "R13"}:
                rows.append(self._aggregate_experiment_workload(outputs_root, code))
            elif code in {"R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"}:
                rows.append(
                    self._real_sensitivity_workload(
                        outputs_root,
                        code,
                        expected_datasets={"nbaiot"},
                        expected_model_seeds=_PRIMARY_MODEL_SEEDS,
                        expected_calibration_seed=1000,
                    )
                )
            elif code == "R12":
                rows.append(self._source_order_workload(outputs_root))
            else:
                rows.append(
                    ExperimentCompletion(
                        protocol_code=code,
                        complete=False,
                        expected_cells=None,
                        observed_cells=0,
                        problems=("no workload reconciliation rule is implemented",),
                    )
                )
        return tuple(rows)

    def _completed_runs(self, runs_root: Path) -> tuple[tuple[RunManifest, Path], ...]:
        rows: list[tuple[RunManifest, Path]] = []
        if not runs_root.exists():
            return ()
        for root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            path = RunLayout(root).manifest
            if not path.is_file():
                continue
            manifest = self.manifests.load(path)
            if manifest.status is ExperimentStatus.COMPLETE:
                rows.append((manifest, root))
        return tuple(rows)

    def _policy_run_workload(
        self,
        code: str,
        experiment_id: ExperimentId,
        runs: tuple[tuple[RunManifest, Path], ...],
        model_seeds: tuple[int, ...],
        calibration_seeds: tuple[int, ...],
        policies: tuple[PolicyId, ...],
        expected_clients: int,
    ) -> ExperimentCompletion:
        selected = [
            (manifest, root) for manifest, root in runs if manifest.experiment_id is experiment_id
        ]
        expected_identities = set(product(model_seeds, calibration_seeds, policies))
        observed_identities = {
            (manifest.model_seed, manifest.calibration_seed, manifest.policy_id)
            for manifest, _ in selected
        }
        problems = self._identity_problems(expected_identities, observed_identities)
        for manifest, root in selected:
            problems.extend(self._validate_metric_rows(manifest, root, expected_clients))
        return ExperimentCompletion(
            protocol_code=code,
            complete=not problems,
            expected_cells=len(expected_identities),
            observed_cells=len(selected),
            problems=tuple(problems),
        )

    def _external_policy_workload(
        self,
        code: str,
        experiment_id: ExperimentId,
        runs: tuple[tuple[RunManifest, Path], ...],
        model_seeds: tuple[int, ...],
        calibration_seeds: tuple[int, ...],
        policies: tuple[PolicyId, ...],
    ) -> ExperimentCompletion:
        selected = [
            (manifest, root) for manifest, root in runs if manifest.experiment_id is experiment_id
        ]
        expected_identities = set(product(model_seeds, calibration_seeds, policies))
        observed_identities = {
            (manifest.model_seed, manifest.calibration_seed, manifest.policy_id)
            for manifest, _ in selected
        }
        problems = self._identity_problems(expected_identities, observed_identities)
        client_counts: set[int] = set()
        for manifest, root in selected:
            record_path = RunLayout(root).metric_records
            if not record_path.is_file():
                problems.append(f"{manifest.run_id.value}: missing metric records")
                continue
            line_count = sum(
                1 for line in record_path.read_text(encoding="utf-8").splitlines() if line
            )
            client_counts.add(line_count)
            if line_count <= 0:
                problems.append(f"{manifest.run_id.value}: no eligible-client metric rows")
        if len(client_counts) > 1:
            problems.append("eligible client count changes across policy cells")
        return ExperimentCompletion(
            protocol_code=code,
            complete=not problems,
            expected_cells=len(expected_identities),
            observed_cells=len(selected),
            problems=tuple(problems),
        )

    @staticmethod
    def _identity_problems(
        expected: set[tuple[int, int, PolicyId]],
        observed: set[tuple[int, int, PolicyId]],
    ) -> list[str]:
        problems: list[str] = []
        missing = expected - observed
        unexpected = observed - expected
        if missing:
            problems.append(f"missing policy-cell identities: {len(missing)}")
        if unexpected:
            problems.append(f"unexpected policy-cell identities: {len(unexpected)}")
        return problems

    @staticmethod
    def _validate_metric_rows(
        manifest: RunManifest,
        root: Path,
        expected_clients: int,
    ) -> list[str]:
        record_path = RunLayout(root).metric_records
        if not record_path.is_file():
            return [f"{manifest.run_id.value}: missing metric records"]
        line_count = sum(1 for line in record_path.read_text(encoding="utf-8").splitlines() if line)
        if line_count != expected_clients:
            return [f"{manifest.run_id.value}: {line_count} client rows != {expected_clients}"]
        return []

    @staticmethod
    def _aggregate_experiment_workload(
        outputs_root: Path,
        code: str,
    ) -> ExperimentCompletion:
        result_path = outputs_root / "experiments" / code / "results.json"
        if not result_path.is_file():
            return ExperimentCompletion(code, False, None, 0, ("experiment evidence missing",))
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        problems: list[str] = []
        if payload.get("experiment") != code:
            problems.append("experiment code mismatch")
        if payload.get("complete") is not True:
            problems.append("experiment result envelope is not complete")
        observed = int(payload.get("observed_cells", 0))
        expected = payload.get("expected_cells")
        expected_int = int(expected) if isinstance(expected, int) else None
        expected_by_code = {
            "S1": 32,
            "S2": 36,
            "S3": 12,
            "S4": 5,
            "S5": 12,
            "S6": 45,
            "R13": 4,
        }[code]
        if expected_int != expected_by_code or observed != expected_by_code:
            problems.append(f"cell ledger {observed}/{expected_int} != roadmap {expected_by_code}")
        expected_trials = {
            "S1": 320_000,
            "S2": 360_000,
            "S3": 120_000,
            "S4": 50_000,
            "S5": 120_000,
            "S6": 0,
            "R13": 0,
        }[code]
        if int(payload.get("observed_monte_carlo_trials", 0)) != expected_trials:
            problems.append("Monte-Carlo trial ledger does not match the roadmap")
        return ExperimentCompletion(
            protocol_code=code,
            complete=not problems,
            expected_cells=expected_by_code,
            observed_cells=observed,
            problems=tuple(problems),
        )

    @staticmethod
    def _real_sensitivity_workload(
        outputs_root: Path,
        code: str,
        expected_datasets: set[str],
        expected_model_seeds: tuple[int, ...],
        expected_calibration_seed: int,
    ) -> ExperimentCompletion:
        cells_root = outputs_root / "experiments" / code / "cells"
        files = tuple(sorted(cells_root.glob("*.json"))) if cells_root.exists() else ()
        problems: list[str] = []
        identities: set[tuple[str, int, int]] = set()
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("experiment") != code:
                problems.append(f"{path.name}: experiment code mismatch")
                continue
            if payload.get("complete") is not True:
                problems.append(f"{path.name}: incomplete subcell ledger")
            dataset = str(payload.get("dataset_id"))
            model_seed = int(payload.get("model_seed", -1))
            calibration_seed = int(payload.get("calibration_seed", -1))
            identities.add((dataset, model_seed, calibration_seed))
        expected = {
            (dataset, model_seed, expected_calibration_seed)
            for dataset in expected_datasets
            for model_seed in expected_model_seeds
        }
        missing = expected - identities
        unexpected = identities - expected
        if missing:
            problems.append(f"missing sensitivity model cells: {len(missing)}")
        if unexpected:
            problems.append(f"unexpected sensitivity model cells: {len(unexpected)}")
        return ExperimentCompletion(
            protocol_code=code,
            complete=not problems,
            expected_cells=len(expected),
            observed_cells=len(files),
            problems=tuple(problems),
        )

    @staticmethod
    def _source_order_workload(outputs_root: Path) -> ExperimentCompletion:
        cells_root = outputs_root / "experiments" / "R12" / "cells"
        files = tuple(sorted(cells_root.glob("*.json"))) if cells_root.exists() else ()
        problems: list[str] = []
        identities: set[tuple[str, int, int]] = set()
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("experiment") != "R12":
                problems.append(f"{path.name}: experiment code mismatch")
                continue
            if payload.get("complete") is not True:
                problems.append(f"{path.name}: source-order evidence incomplete")
            identities.add(
                (
                    str(payload.get("dataset_id")),
                    int(payload.get("model_seed", -1)),
                    int(payload.get("calibration_seed", -1)),
                )
            )
        expected = {
            *(("nbaiot", model_seed, 1000) for model_seed in _PRIMARY_MODEL_SEEDS),
            *(("diad", model_seed, 2000) for model_seed in _PRIMARY_MODEL_SEEDS),
        }
        missing = expected - identities
        unexpected = identities - expected
        if missing:
            problems.append(f"missing source-order model cells: {len(missing)}")
        if unexpected:
            problems.append(f"unexpected source-order model cells: {len(unexpected)}")
        return ExperimentCompletion(
            protocol_code="R12",
            complete=not problems,
            expected_cells=len(expected),
            observed_cells=len(files),
            problems=tuple(problems),
        )
