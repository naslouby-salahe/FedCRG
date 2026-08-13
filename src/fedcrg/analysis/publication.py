"""Build the pre-registered publication tables and figures from frozen evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import pandas as pd

from fedcrg.analysis.figures import (
    assumption_stress,
    calibration_phase_transition,
    decision_architecture,
    external_replication,
    mismatch_power_map,
    per_client_operating_points,
    readiness_frontier,
    reliability_utility_frontier,
)
from fedcrg.analysis.tables import PublicationTableBuilder
from fedcrg.config.models import ExperimentConfig
from fedcrg.core.enums import PolicyId


@dataclass(frozen=True, slots=True)
class PublicationArtifact:
    name: str
    path: Path | None
    available: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PublicationPackage:
    tables: tuple[PublicationArtifact, ...]
    figures: tuple[PublicationArtifact, ...]
    manifest: Path

    @property
    def complete(self) -> bool:
        return all(item.available for item in (*self.tables, *self.figures))


class PublicationPackageBuilder:
    """Generate Tables 1–8 and Figures 1–8 without manually entered outcomes."""

    def __init__(self, tables: PublicationTableBuilder | None = None) -> None:
        self.tables = tables or PublicationTableBuilder()

    def build(
        self,
        *,
        config: ExperimentConfig,
        outputs_root: Path,
        prepared_manifest: Path | None = None,
        destination: Path | None = None,
    ) -> PublicationPackage:
        destination = destination or outputs_root / "reports" / "publication"
        table_root = destination / "tables"
        figure_root = destination / "figures"
        table_root.mkdir(parents=True, exist_ok=True)
        figure_root.mkdir(parents=True, exist_ok=True)

        runs = tuple(self._completed_runs(outputs_root / "runs"))
        primary_runs = tuple(
            path for path in runs if self._manifest_value(path, "experiment_id") == "primary_nbaiot"
        )
        external_runs = tuple(
            path for path in runs if self._manifest_value(path, "experiment_id") == "external_diad"
        )
        named_primary = tuple(
            path for path in primary_runs if self._manifest_value(path, "calibration_seed") == 1000
        )
        named_external = tuple(
            path for path in external_runs if self._manifest_value(path, "calibration_seed") == 2000
        )

        tables = (
            self._table(
                "Table 1 - Literature boundary",
                lambda: self.tables.literature_boundary(
                    table_root / "table_1_literature_boundary.csv"
                ),
            ),
            self._table(
                "Table 2 - Protocol constants",
                lambda: self.tables.protocol_constants(
                    config, table_root / "table_2_protocol_constants.csv"
                ),
            ),
            self._optional_table(
                "Table 3 - Dataset inventory",
                prepared_manifest,
                lambda path: self.tables.dataset_inventory(
                    path, table_root / "table_3_dataset_inventory.csv"
                ),
                "prepared dataset manifest is unavailable",
            ),
            self._runs_table(
                "Table 4 - Primary policy results",
                primary_runs,
                lambda: self.tables.primary_policy_results(
                    primary_runs, table_root / "table_4_primary_policy_results.csv"
                ),
                "R1 policy runs are unavailable",
            ),
            self._runs_table(
                "Table 5 - Admission states",
                tuple(
                    path
                    for path in named_primary
                    if self._manifest_value(path, "policy_id") == PolicyId.FEDCRG.value
                ),
                lambda: self.tables.admission_states_from_runs(
                    tuple(
                        path
                        for path in named_primary
                        if self._manifest_value(path, "policy_id") == PolicyId.FEDCRG.value
                    ),
                    table_root / "table_5_admission_states.csv",
                ),
                "named-split FedCRG threshold records are unavailable",
            ),
            self._runs_table(
                "Table 6 - Ablations",
                primary_runs,
                lambda: self.tables.ablations(primary_runs, table_root / "table_6_ablations.csv"),
                "R1 ablation runs are unavailable",
            ),
            self._experiment_table(
                "Table 7 - Sensitivity",
                outputs_root / "experiments",
                lambda: self.tables.sensitivity(
                    outputs_root / "experiments",
                    table_root / "table_7_sensitivity.csv",
                ),
                "R2-R7 sensitivity evidence is unavailable",
                required_codes=("R2", "R3", "R4", "R5", "R6", "R7"),
            ),
            self._runs_table(
                "Table 8 - External replication",
                external_runs,
                lambda: self.tables.external_replication(
                    external_runs, table_root / "table_8_external_replication.csv"
                ),
                "R10 external-replication runs are unavailable",
            ),
        )

        figures = (
            self._figure(
                "Figure 1 - Decision architecture",
                lambda: decision_architecture(figure_root / "figure_1_decision_architecture.png"),
            ),
            self._frame_figure(
                "Figure 2 - Readiness frontier",
                self._readiness_frame(outputs_root),
                lambda frame: readiness_frontier(
                    frame, figure_root / "figure_2_readiness_frontier.png"
                ),
                "precomputed readiness evidence is unavailable",
            ),
            self._frame_figure(
                "Figure 3 - Mismatch evidence power",
                self._experiment_cells(outputs_root, "S6"),
                lambda frame: mismatch_power_map(
                    frame, figure_root / "figure_3_mismatch_power.png"
                ),
                "S6 exact power evidence is unavailable",
            ),
            self._frame_figure(
                "Figure 4 - Per-client operating points",
                self._client_operating_frame(named_primary),
                lambda frame: per_client_operating_points(
                    frame, figure_root / "figure_4_client_operating_points.png"
                ),
                "named-split R1 client metrics are unavailable",
            ),
            self._frame_figure(
                "Figure 5 - Reliability-utility frontier",
                self._reliability_utility_frame(named_primary),
                lambda frame: reliability_utility_frontier(
                    frame, figure_root / "figure_5_reliability_utility.png"
                ),
                "named-split R1 federation metrics are unavailable",
            ),
            self._frame_figure(
                "Figure 6 - Evidence-size phase transition",
                self._phase_transition_frame(outputs_root),
                lambda frame: calibration_phase_transition(
                    frame, figure_root / "figure_6_phase_transition.png"
                ),
                "R2/R3 evidence-size sensitivity is unavailable",
            ),
            self._frame_figure(
                "Figure 7 - Assumption stress",
                self._assumption_stress_frame(outputs_root),
                lambda frame: assumption_stress(
                    frame, figure_root / "figure_7_assumption_stress.png"
                ),
                "S3-S5 assumption-stress evidence is unavailable",
            ),
            self._frame_figure(
                "Figure 8 - External replication",
                self._external_client_frame(named_external),
                lambda frame: external_replication(
                    frame, figure_root / "figure_8_external_replication.png"
                ),
                "named-split R10 client metrics are unavailable",
            ),
        )

        manifest = destination / "manifest.json"
        payload = {
            "complete": all(item.available for item in (*tables, *figures)),
            "tables": [self._artifact_dict(item, destination) for item in tables],
            "figures": [self._artifact_dict(item, destination) for item in figures],
            "source_policy": "all numerical content is derived from immutable FedCRG artifacts",
        }
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return PublicationPackage(tables, figures, manifest)

    @staticmethod
    def _artifact_dict(item: PublicationArtifact, root: Path) -> dict[str, object]:
        return {
            "name": item.name,
            "available": item.available,
            "path": None if item.path is None else item.path.relative_to(root).as_posix(),
            "reason": item.reason,
        }

    @staticmethod
    def _table(name: str, build: Callable[[], Path]) -> PublicationArtifact:
        try:
            return PublicationArtifact(name, build(), True)
        except Exception as exc:
            return PublicationArtifact(name, None, False, f"{type(exc).__name__}: {exc}")

    @classmethod
    def _figure(cls, name: str, build: Callable[[], Path]) -> PublicationArtifact:
        return cls._table(name, build)

    @classmethod
    def _optional_table(
        cls,
        name: str,
        source: Path | None,
        build: Callable[[Path], Path],
        reason: str,
    ) -> PublicationArtifact:
        if source is None or not source.is_file():
            return PublicationArtifact(name, None, False, reason)
        return cls._table(name, lambda: build(source))

    @classmethod
    def _runs_table(
        cls,
        name: str,
        runs: tuple[Path, ...],
        build: Callable[[], Path],
        reason: str,
    ) -> PublicationArtifact:
        if not runs:
            return PublicationArtifact(name, None, False, reason)
        return cls._table(name, build)

    @classmethod
    def _experiment_table(
        cls,
        name: str,
        root: Path,
        build: Callable[[], Path],
        reason: str,
        required_codes: tuple[str, ...],
    ) -> PublicationArtifact:
        if not all((root / code).exists() for code in required_codes):
            return PublicationArtifact(name, None, False, reason)
        return cls._table(name, build)

    @classmethod
    def _frame_figure(
        cls,
        name: str,
        frame: pd.DataFrame,
        build: Callable[[pd.DataFrame], Path],
        reason: str,
    ) -> PublicationArtifact:
        if frame.empty:
            return PublicationArtifact(name, None, False, reason)
        return cls._figure(name, lambda: build(frame))

    @staticmethod
    def _completed_runs(root: Path):
        if not root.exists():
            return
        for path in sorted(item for item in root.iterdir() if item.is_dir()):
            manifest = path / "manifest.json"
            if not manifest.is_file():
                continue
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("status") == "complete":
                yield path

    @staticmethod
    def _manifest_value(run: Path, key: str) -> object:
        payload = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        return payload.get(key)

    @staticmethod
    def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
        if not path.is_file():
            return ()
        return tuple(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )

    @staticmethod
    def _readiness_frame(outputs_root: Path) -> pd.DataFrame:
        path = outputs_root / "cache" / "precomputed" / "readiness_plans.json"
        if not path.is_file():
            return pd.DataFrame()
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = [
            row
            for row in payload.get("entries", [])
            if abs(float(row.get("a", -1.0)) - 0.005) < 1e-15
            and abs(float(row.get("b", -1.0)) - 0.015) < 1e-15
            and abs(float(row.get("assurance", -1.0)) - 0.95) < 1e-15
        ]
        return pd.DataFrame.from_records(rows).sort_values("n") if rows else pd.DataFrame()

    @staticmethod
    def _experiment_cells(outputs_root: Path, code: str) -> pd.DataFrame:
        path = outputs_root / "experiments" / code / "results.json"
        if not path.is_file():
            return pd.DataFrame()
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells = payload.get("cells", [])
        return pd.DataFrame.from_records(cells if isinstance(cells, list) else [])

    @classmethod
    def _client_operating_frame(cls, runs: tuple[Path, ...]) -> pd.DataFrame:
        selected = {
            PolicyId.GLOBAL_QUANTILE.value,
            PolicyId.LOCAL_QUANTILE.value,
            PolicyId.SHRINKAGE.value,
            PolicyId.FEDCRG.value,
        }
        rows: list[dict[str, object]] = []
        for run in runs:
            policy = str(cls._manifest_value(run, "policy_id"))
            if policy not in selected:
                continue
            for row in cls._jsonl(run / "metrics" / "metric_record.jsonl"):
                rows.append({"client_id": row["client_id"], "policy": policy, "fpr": row["fpr"]})
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame.from_records(rows)
        return frame.groupby(["client_id", "policy"], as_index=False)["fpr"].mean()

    @classmethod
    def _reliability_utility_frame(cls, runs: tuple[Path, ...]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for run in runs:
            path = run / "metrics" / "federation.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "policy": cls._manifest_value(run, "policy_id"),
                    "mebe": payload.get("mebe"),
                    "attack_balanced_macro_tpr": payload.get("attack_balanced_macro_tpr"),
                }
            )
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame.from_records(rows).dropna(subset=["mebe", "attack_balanced_macro_tpr"])
        if frame.empty:
            return frame
        return frame.groupby("policy", as_index=False)[["mebe", "attack_balanced_macro_tpr"]].mean()

    @staticmethod
    def _sensitivity_cell_files(outputs_root: Path, code: str) -> tuple[Path, ...]:
        root = outputs_root / "experiments" / code / "cells"
        if not root.exists():
            return ()
        return tuple(sorted(root.glob("*.json")))

    @classmethod
    def _phase_transition_frame(cls, outputs_root: Path) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for code, evidence, key in (
            ("R2", "readiness", "sample_count"),
            ("R3", "mismatch", "sample_count"),
        ):
            for path in cls._sensitivity_cell_files(outputs_root, code):
                payload = json.loads(path.read_text(encoding="utf-8"))
                for cell in payload.get("cells", []):
                    if not isinstance(cell, dict) or key not in cell:
                        continue
                    evaluation = cell.get("evaluation")
                    if not isinstance(evaluation, dict):
                        continue
                    protocol_rows = evaluation.get("protocol_results")
                    if not isinstance(protocol_rows, list) or not protocol_rows:
                        continue
                    admitted = sum(
                        1
                        for item in protocol_rows
                        if isinstance(item, dict)
                        and item.get("decision_state") == "LOCAL_PERSONALIZE"
                    )
                    rows.append(
                        {
                            "evidence": evidence,
                            "sample_count": int(cell[key]),
                            "admission_rate": admitted / len(protocol_rows),
                        }
                    )
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame.from_records(rows)
            .groupby(["evidence", "sample_count"], as_index=False)["admission_rate"]
            .mean()
        )

    @classmethod
    def _assumption_stress_frame(cls, outputs_root: Path) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for code, condition_name in (
            ("S3", "temporal_dependence"),
            ("S4", "mean_shift"),
            ("S5", "contamination"),
        ):
            frame = cls._experiment_cells(outputs_root, code)
            if frame.empty:
                continue
            for _, row in frame.iterrows():
                if code == "S3":
                    value = row.get("value")
                    coverage = row.get("coverage")
                elif code == "S4":
                    value = row.get("value")
                    coverage = row.get("coverage")
                else:
                    value = row.get("fraction")
                    coverage = row.get("empirical_probability")
                if value is None or coverage is None:
                    continue
                label = condition_name
                if code == "S5" and row.get("direction") is not None:
                    label = f"{condition_name}_{row.get('direction')}"
                rows.append({"condition": label, "value": value, "coverage": coverage})
        return pd.DataFrame.from_records(rows)

    @classmethod
    def _external_client_frame(cls, runs: tuple[Path, ...]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for run in runs:
            policy = str(cls._manifest_value(run, "policy_id"))
            for row in cls._jsonl(run / "metrics" / "metric_record.jsonl"):
                rows.append({"client_id": row["client_id"], "policy": policy, "fpr": row["fpr"]})
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame.from_records(rows)
        return frame.groupby(["client_id", "policy"], as_index=False)["fpr"].mean()
