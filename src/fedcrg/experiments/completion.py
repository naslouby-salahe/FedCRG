"""Pre-registered workload reconciliation for generated experiment evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from fedcrg.artifacts.manifests import RunManifest, RunManifestStore
from fedcrg.artifacts.paths import RunLayout
from fedcrg.domain.constants import (
    DEEP_SVDD_MODEL_SEEDS,
    DIAD_NAMED_CALIBRATION_SEED,
    NBAIOT_NAMED_CALIBRATION_SEED,
    PRIMARY_MODEL_SEEDS,
)
from fedcrg.domain.enums import DatasetId, ExperimentId, ExperimentStatus, PolicyId
from fedcrg.experiments.experiment_definition import (
    SECOND_DETECTOR_POLICIES,
    all_experiment_definitions,
)

_PRIMARY_MODEL_SEEDS = PRIMARY_MODEL_SEEDS
_NBAIOT_CALIBRATION_SEEDS = tuple(
    range(NBAIOT_NAMED_CALIBRATION_SEED, NBAIOT_NAMED_CALIBRATION_SEED + 50)
)
_DIAD_CALIBRATION_SEEDS = tuple(
    range(DIAD_NAMED_CALIBRATION_SEED, DIAD_NAMED_CALIBRATION_SEED + 20)
)
_SECOND_DETECTOR_MODEL_SEEDS = DEEP_SVDD_MODEL_SEEDS
_SECOND_DETECTOR_CALIBRATION_SEEDS = tuple(
    range(NBAIOT_NAMED_CALIBRATION_SEED, NBAIOT_NAMED_CALIBRATION_SEED + 10)
)

_AGGREGATE_WORKLOAD_EXPERIMENTS = (
    ExperimentId.READINESS_THEOREM,
    ExperimentId.TARGET_FPR_SYNTHETIC,
    ExperimentId.TEMPORAL_DEPENDENCE,
    ExperimentId.CALIBRATION_SHIFT,
    ExperimentId.CALIBRATION_CONTAMINATION,
    ExperimentId.MISMATCH_POWER,
    ExperimentId.COMPUTATIONAL_BENCHMARK,
)
_REAL_SENSITIVITY_EXPERIMENTS = (
    ExperimentId.READINESS_SAMPLE_SIZE,
    ExperimentId.MISMATCH_SAMPLE_SIZE,
    ExperimentId.TOLERANCE_SENSITIVITY,
    ExperimentId.TARGET_FPR_REAL,
    ExperimentId.ASSURANCE_SENSITIVITY,
    ExperimentId.REAL_CONTAMINATION,
)
_SINGLE_SEED_SENSITIVITY_EXPERIMENTS = (
    ExperimentId.MULTIPLICITY_SENSITIVITY,
    ExperimentId.SOURCE_ORDER_TEST,
)
_EXPECTED_CELLS_BY_EXPERIMENT = {
    ExperimentId.READINESS_THEOREM: 32,
    ExperimentId.TARGET_FPR_SYNTHETIC: 36,
    ExperimentId.TEMPORAL_DEPENDENCE: 12,
    ExperimentId.CALIBRATION_SHIFT: 5,
    ExperimentId.CALIBRATION_CONTAMINATION: 12,
    ExperimentId.MISMATCH_POWER: 45,
    ExperimentId.COMPUTATIONAL_BENCHMARK: 4,
}
_EXPECTED_MONTE_CARLO_TRIALS_BY_EXPERIMENT = {
    ExperimentId.READINESS_THEOREM: 320_000,
    ExperimentId.TARGET_FPR_SYNTHETIC: 360_000,
    ExperimentId.TEMPORAL_DEPENDENCE: 120_000,
    ExperimentId.CALIBRATION_SHIFT: 50_000,
    ExperimentId.CALIBRATION_CONTAMINATION: 120_000,
    ExperimentId.MISMATCH_POWER: 0,
    ExperimentId.COMPUTATIONAL_BENCHMARK: 0,
}


@dataclass(frozen=True, slots=True)
class ExperimentCompletion:
    experiment_id: ExperimentId
    complete: bool
    expected_cells: int | None
    observed_cells: int
    problems: tuple[str, ...]


class ExperimentCompletionAuditor:
    """Reconcile every pre-registered workload without inferring missing evidence."""

    def __init__(self) -> None:
        self.manifests = RunManifestStore()

    def audit(self, outputs_root: Path) -> tuple[ExperimentCompletion, ...]:
        runs = self._completed_runs(outputs_root / "runs")
        rows: list[ExperimentCompletion] = []
        for definition in all_experiment_definitions():
            experiment_id = definition.id
            if experiment_id is ExperimentId.PRIMARY_NBAIOT:
                rows.append(
                    self._policy_run_workload(
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
                        experiment_id,
                        runs,
                        _PRIMARY_MODEL_SEEDS,
                        (DIAD_NAMED_CALIBRATION_SEED,),
                        definition.policies,
                    )
                )
            elif experiment_id in _AGGREGATE_WORKLOAD_EXPERIMENTS:
                rows.append(self._aggregate_experiment_workload(outputs_root, experiment_id))
            elif experiment_id in _REAL_SENSITIVITY_EXPERIMENTS:
                rows.append(
                    self._real_sensitivity_workload(
                        outputs_root,
                        experiment_id,
                        expected_model_seeds=_PRIMARY_MODEL_SEEDS,
                        expected_calibration_seed=NBAIOT_NAMED_CALIBRATION_SEED,
                    )
                )
            elif experiment_id in _SINGLE_SEED_SENSITIVITY_EXPERIMENTS:
                rows.append(
                    self._single_seed_sensitivity_workload(
                        outputs_root,
                        experiment_id,
                        expected_calibration_seed=NBAIOT_NAMED_CALIBRATION_SEED,
                    )
                )
            elif experiment_id is ExperimentId.SOURCE_ORDER_CALIBRATION:
                rows.append(self._source_order_workload(outputs_root))
            else:
                rows.append(
                    ExperimentCompletion(
                        experiment_id=experiment_id,
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
            (int(manifest.model_seed), int(manifest.calibration_seed), manifest.policy_id)
            for manifest, _ in selected
        }
        problems = self._identity_problems(expected_identities, observed_identities)
        for manifest, root in selected:
            problems.extend(self._validate_metric_rows(manifest, root, expected_clients))
        return ExperimentCompletion(
            experiment_id=experiment_id,
            complete=not problems,
            expected_cells=len(expected_identities),
            observed_cells=len(selected),
            problems=tuple(problems),
        )

    def _external_policy_workload(
        self,
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
            (int(manifest.model_seed), int(manifest.calibration_seed), manifest.policy_id)
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
            experiment_id=experiment_id,
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
        experiment_id: ExperimentId,
    ) -> ExperimentCompletion:
        result_path = outputs_root / "experiments" / experiment_id.value / "results.json"
        if not result_path.is_file():
            return ExperimentCompletion(
                experiment_id, False, None, 0, ("experiment evidence missing",)
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        problems: list[str] = []
        if payload.get("experiment") != experiment_id.value:
            problems.append("experiment identity mismatch")
        if payload.get("complete") is not True:
            problems.append("experiment result envelope is not complete")
        observed = int(payload.get("observed_cells", 0))
        expected = payload.get("expected_cells")
        expected_int = int(expected) if isinstance(expected, int) else None
        expected_by_experiment = _EXPECTED_CELLS_BY_EXPERIMENT[experiment_id]
        if expected_int != expected_by_experiment or observed != expected_by_experiment:
            problems.append(
                f"cell ledger {observed}/{expected_int} != pre-registered {expected_by_experiment}"
            )
        expected_trials = _EXPECTED_MONTE_CARLO_TRIALS_BY_EXPERIMENT[experiment_id]
        if int(payload.get("observed_monte_carlo_trials", 0)) != expected_trials:
            problems.append("Monte-Carlo trial ledger does not match the pre-registered count")
        return ExperimentCompletion(
            experiment_id=experiment_id,
            complete=not problems,
            expected_cells=expected_by_experiment,
            observed_cells=observed,
            problems=tuple(problems),
        )

    @staticmethod
    def _real_sensitivity_workload(
        outputs_root: Path,
        experiment_id: ExperimentId,
        expected_model_seeds: tuple[int, ...],
        expected_calibration_seed: int,
    ) -> ExperimentCompletion:
        """Reconcile a SensitivityEnvelope (experiment_id/model_seed/calibration_seed/cells)."""
        cells_root = outputs_root / "experiments" / experiment_id.value / "cells"
        files = tuple(sorted(cells_root.glob("*.json"))) if cells_root.exists() else ()
        problems: list[str] = []
        identities: set[tuple[int, int]] = set()
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("experiment_id") != experiment_id.value:
                problems.append(f"{path.name}: experiment identity mismatch")
                continue
            cells = payload.get("cells")
            if not isinstance(cells, list) or not cells:
                problems.append(f"{path.name}: sensitivity envelope has no cells")
                continue
            model_seed = int(payload.get("model_seed", -1))
            calibration_seed = int(payload.get("calibration_seed", -1))
            identities.add((model_seed, calibration_seed))
        expected = {(model_seed, expected_calibration_seed) for model_seed in expected_model_seeds}
        missing = expected - identities
        unexpected = identities - expected
        if missing:
            problems.append(f"missing sensitivity model cells: {len(missing)}")
        if unexpected:
            problems.append(f"unexpected sensitivity model cells: {len(unexpected)}")
        return ExperimentCompletion(
            experiment_id=experiment_id,
            complete=not problems,
            expected_cells=len(expected),
            observed_cells=len(files),
            problems=tuple(problems),
        )

    @staticmethod
    def _single_seed_sensitivity_workload(
        outputs_root: Path,
        experiment_id: ExperimentId,
        expected_calibration_seed: int,
    ) -> ExperimentCompletion:
        """Reconcile a MultiplicityEnvelope/SourceOrderEnvelope: no model-seed axis."""
        cells_root = outputs_root / "experiments" / experiment_id.value / "cells"
        files = tuple(sorted(cells_root.glob("*.json"))) if cells_root.exists() else ()
        problems: list[str] = []
        identities: set[int] = set()
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("experiment_id") != experiment_id.value:
                problems.append(f"{path.name}: experiment identity mismatch")
                continue
            cells = payload.get("cells")
            if not isinstance(cells, list) or not cells:
                problems.append(f"{path.name}: sensitivity envelope has no cells")
                continue
            identities.add(int(payload.get("calibration_seed", -1)))
        expected = {expected_calibration_seed}
        missing = expected - identities
        unexpected = identities - expected
        if missing:
            problems.append(f"missing sensitivity calibration seeds: {len(missing)}")
        if unexpected:
            problems.append(f"unexpected sensitivity calibration seeds: {len(unexpected)}")
        return ExperimentCompletion(
            experiment_id=experiment_id,
            complete=not problems,
            expected_cells=len(expected),
            observed_cells=len(files),
            problems=tuple(problems),
        )

    @staticmethod
    def _source_order_workload(outputs_root: Path) -> ExperimentCompletion:
        experiment_id = ExperimentId.SOURCE_ORDER_CALIBRATION
        cells_root = outputs_root / "experiments" / experiment_id.value / "cells"
        files = tuple(sorted(cells_root.glob("*.json"))) if cells_root.exists() else ()
        problems: list[str] = []
        identities: set[tuple[str, int, int]] = set()
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("experiment") != experiment_id.value:
                problems.append(f"{path.name}: experiment identity mismatch")
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
            *(
                (DatasetId.NBAIOT.value, model_seed, NBAIOT_NAMED_CALIBRATION_SEED)
                for model_seed in _PRIMARY_MODEL_SEEDS
            ),
            *(
                (DatasetId.DIAD.value, model_seed, DIAD_NAMED_CALIBRATION_SEED)
                for model_seed in _PRIMARY_MODEL_SEEDS
            ),
        }
        missing = expected - identities
        unexpected = identities - expected
        if missing:
            problems.append(f"missing source-order model cells: {len(missing)}")
        if unexpected:
            problems.append(f"unexpected source-order model cells: {len(unexpected)}")
        return ExperimentCompletion(
            experiment_id=experiment_id,
            complete=not problems,
            expected_cells=len(expected),
            observed_cells=len(files),
            problems=tuple(problems),
        )
