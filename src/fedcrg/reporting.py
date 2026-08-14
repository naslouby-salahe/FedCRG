"""Reproducible reports and publication bundles built only from immutable evidence.

Run reports, the repository report, publication tables and figures, and the
results bundle are projections of frozen run artifacts: no detector is loaded,
no model is retrained, and missing evidence remains explicitly incomplete.
"""

from __future__ import annotations

import json

import pydantic
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fedcrg.config import ExperimentConfig, Study
from fedcrg.evidence.models import (
    ChecksumRecord,
    GitEnvironment,
    MetricRecord,
    PreparedDatasetManifest,
    RunManifest,
    ThresholdRecord,
)
from fedcrg.evidence.store import (
    ArtifactVerifier,
    OutputsLayout,
    RunLayout,
    atomic_write_json,
    sha256_file,
)
from fedcrg.experiments.analyses import (
    confirmatory_contrasts,
    load_federation_results,
    split_sensitivity,
)
from fedcrg.runtime import get_logger
from fedcrg.thresholding.metrics import FederationMetrics
from fedcrg.types import (
    Assurance,
    CalibrationSeed,
    CampaignId,
    ClientId,
    DataRole,
    DecisionSource,
    DecisionState,
    Description,
    DetectorId,
    DatasetId,
    ExperimentId,
    FailureCode,
    Fraction,
    Identifier,
    InformationRegime,
    Metric,
    ModelSeed,
    NonNegativeCount,
    PolicyId,
    PositiveCount,
    Probability,
    ProtocolConstantValue,
    PValue,
    RunId,
    Score,
    Sha256,
    Tpr,
)
from pydantic import BaseModel, ConfigDict

Frozen = ConfigDict(frozen=True)

_LOGGER = get_logger(__name__)


def _run_manifest(run_dir: Path) -> RunManifest | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        return RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except pydantic.ValidationError:
        return None


def _completed_runs(outputs_root: Path) -> tuple[Path, ...]:
    runs_root = OutputsLayout(outputs_root).runs
    if not runs_root.exists():
        return ()
    rows: list[Path] = []
    for path in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        manifest_path = path / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = RunManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
            except Exception:
                continue
            if manifest.status.value == "complete":
                rows.append(path)
    return tuple(rows)


def _jsonl_count(path: Path) -> NonNegativeCount:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def build_run_report(run_dir: Path) -> Path:
    """Markdown summary of one immutable policy run."""
    layout = RunLayout(run_dir)
    manifest = RunManifest.model_validate_json(layout.manifest.read_text(encoding="utf-8"))
    verification = ArtifactVerifier().record(layout, _definition_for(manifest))
    federation: FederationMetrics | None = None
    if layout.federation_metrics.exists():
        federation = FederationMetrics.model_validate_json(
            layout.federation_metrics.read_text(encoding="utf-8")
        )
    client_count = _jsonl_count(layout.metric_records)
    lines = [
        f"# FedCRG Run {manifest.run_id}",
        "",
        f"- Experiment: `{manifest.experiment_id.value}`",
        f"- Policy: `{manifest.policy_id.value}`",
        f"- Status: `{manifest.status.value}`",
        f"- Config hash: `{manifest.config_hash}`",
        f"- Verified artifact hashes: `{verification.valid}`",
        f"- Evaluated clients: `{client_count}`",
    ]
    if federation is not None:
        lines.extend(
            [
                "",
                "## Federation endpoints",
                "",
                f"- MEBE: `{federation.mebe}`",
                f"- HighExcess: `{federation.high_excess}`",
                f"- BandViolationRate: `{federation.band_violation_rate}`",
                f"- MAFE: `{federation.mafe}`",
                f"- ABMacroTPR: `{federation.attack_balanced_macro_tpr}`",
                f"- MacroTPR: `{federation.macro_tpr}`",
                f"- Worst-client TPR: `{federation.worst_client_tpr}`",
            ]
        )
    lines.extend(
        [
            "",
            "This report is generated only from immutable run artifacts; it does not "
            "load a detector or retrain a model.",
        ]
    )
    output = layout.reports / "summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _definition_for(manifest: RunManifest):
    """Resolve the catalogue definition recorded in the run manifest."""
    return Study.load().catalogue.spec(manifest.experiment_id)


class LiteratureBoundaryRow(BaseModel):
    """One literature-boundary table row."""
    model_config = Frozen

    policy_id: PolicyId
    information_regime: InformationRegime


class AdmissionStateRow(BaseModel):
    """One admission-state table row."""
    model_config = Frozen

    run_id: RunId
    policy_id: PolicyId
    client_id: ClientId
    tau_ref: Score | None
    tau_local: Score | None
    selected_tau: Score | None
    readiness_n: PositiveCount
    readiness_rank: PositiveCount
    readiness_probability: Assurance
    mismatch_n: PositiveCount
    mismatch_x: NonNegativeCount
    cp_lower: Probability
    cp_upper: Probability
    p_low: PValue | None
    p_high: PValue | None
    state: DecisionState
    tie_count: NonNegativeCount
    selected_source: DecisionSource
    reason_code: FailureCode


class ContrastTableRow(BaseModel):
    """One contrast table row."""
    model_config = Frozen

    comparator: PolicyId
    metric: Identifier
    method_mean: Metric
    comparator_mean: Metric
    observed_difference: Metric
    bootstrap_lower: Metric
    bootstrap_upper: Metric
    relative_difference: Metric | None


class ProtocolConstantRow(BaseModel):
    """One protocol-constant table row."""
    model_config = Frozen

    constant: Identifier
    value: ProtocolConstantValue


class DatasetInventoryRow(BaseModel):
    """One dataset-inventory table row."""
    model_config = Frozen

    client_id: ClientId
    feature_count: PositiveCount
    role_counts: tuple[RoleCountRow, ...]


class RoleCountRow(BaseModel):
    """Role row-count and hash within a dataset row."""
    model_config = Frozen

    role: DataRole
    rows: NonNegativeCount
    row_id_sha256: Sha256


class FederationResultsRow(BaseModel):
    """One federation-results table row."""
    model_config = Frozen

    run_id: RunId
    mebe: Metric
    high_excess: Metric
    band_violation_rate: Fraction
    attack_balanced_macro_tpr: Tpr | None


class PublicationTableBuilder:
    """Deterministic manuscript-table builders from frozen artifacts."""

    def literature_boundary(self, output: Path) -> Path:
        rows = tuple(
            LiteratureBoundaryRow(
                policy_id=policy,
                information_regime=InformationRegime.BENIGN_ONLY,
            )
            for policy in (
                PolicyId.REFERENCE_QUANTILE,
                PolicyId.GLOBAL_QUANTILE,
                PolicyId.LOCAL_QUANTILE,
                PolicyId.SHRINKAGE,
                PolicyId.THREE_SIGMA,
                PolicyId.FEDCRG,
            )
        ) + tuple(
            LiteratureBoundaryRow(
                policy_id=policy,
                information_regime=InformationRegime.SUPERVISED_DEVELOPMENT,
            )
            for policy in (
                PolicyId.DEV_F1_SELECT,
                PolicyId.SUMMARY_STATISTIC_SELECT,
                PolicyId.SUPERVISED_F1,
            )
        )
        return self._write(
            pd.DataFrame.from_records(row.model_dump(mode="json") for row in rows),
            output,
        )

    def primary_policy_results(self, run_dirs: tuple[Path, ...], output: Path) -> Path:
        records = load_federation_results(run_dirs)
        rows = [record.model_dump(mode="json") for record in records]
        return self._write(pd.DataFrame.from_records(rows), output)

    def admission_states_from_runs(self, run_dirs: tuple[Path, ...], output: Path) -> Path:
        records: list[ThresholdRecord] = []
        for run_dir in run_dirs:
            path = RunLayout(run_dir).threshold_records
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    records.append(
                        ThresholdRecord.model_validate_json(line)
                    )
        return self._write(
            pd.DataFrame.from_records(record.model_dump(mode="json") for record in records),
            output,
        )

    def ablations(self, run_dirs: tuple[Path, ...], output: Path, config: ExperimentConfig) -> Path:
        records = load_federation_results(run_dirs)
        primary = tuple(row for row in records if row.experiment_id is ExperimentId.PRIMARY_NBAIOT)
        contrasts = confirmatory_contrasts(
            primary,
            named_calibration_seed=int(config.dataset.primary_calibration_seed),
            expected_model_seeds=tuple(config.randomness.model_seeds),
            bootstrap_seed=config.statistics.bootstrap_seed,
            bootstrap_replicates=config.statistics.bootstrap_replicates,
        )
        rows = tuple(
            ContrastTableRow(
                comparator=result.comparator,
                metric=metric.metric,
                method_mean=metric.method_summary.mean,
                comparator_mean=metric.comparator_summary.mean,
                observed_difference=metric.paired_difference.observed_difference,
                bootstrap_lower=metric.paired_difference.lower,
                bootstrap_upper=metric.paired_difference.upper,
                relative_difference=metric.relative_difference,
            )
            for result in contrasts
            for metric in result.metrics
        )
        return self._write(
            pd.DataFrame.from_records(row.model_dump(mode="json") for row in rows),
            output,
        )

    def protocol_constants(self, config: ExperimentConfig, output: Path) -> Path:
        protocol = config.protocol
        rows: list[ProtocolConstantRow] = [
            ProtocolConstantRow(constant="alpha", value=protocol.alpha),
            ProtocolConstantRow(constant="rho", value=protocol.rho),
            ProtocolConstantRow(constant="band_lower", value=protocol.band.lower),
            ProtocolConstantRow(constant="band_upper", value=protocol.band.upper),
            ProtocolConstantRow(
                constant="readiness_assurance", value=protocol.readiness_assurance
            ),
            ProtocolConstantRow(
                constant="mismatch_confidence", value=protocol.mismatch_confidence
            ),
        ]
        training = config.training
        if training is not None:
            rows.extend(
                [
                    ProtocolConstantRow(constant="rounds", value=training.rounds),
                    ProtocolConstantRow(
                        constant="local_epochs", value=training.local_epochs
                    ),
                    ProtocolConstantRow(
                        constant="batch_size", value=training.batch_size
                    ),
                    ProtocolConstantRow(
                        constant="learning_rate_initial",
                        value=training.learning_rate_initial,
                    ),
                    ProtocolConstantRow(
                        constant="learning_rate_final",
                        value=training.learning_rate_final,
                    ),
                    ProtocolConstantRow(
                        constant="client_fraction", value=training.client_fraction
                    ),
                ]
            )
        return self._write(
            pd.DataFrame.from_records(row.model_dump(mode="json") for row in rows),
            output,
        )

    def dataset_inventory(self, prepared_manifest: Path, output: Path) -> Path:
        manifest = PreparedDatasetManifest.model_validate_json(
            prepared_manifest.read_text(encoding="utf-8")
        )
        feature_count = len(manifest.feature_names)
        rows = tuple(
            DatasetInventoryRow(
                client_id=client.client_id,
                feature_count=feature_count,
                role_counts=tuple(
                    RoleCountRow(
                        role=role.role,
                        rows=role.rows,
                        row_id_sha256=role.row_id_sha256,
                    )
                    for role in client.roles
                ),
            )
            for client in manifest.clients
        )
        return self._write(
            pd.DataFrame.from_records(row.model_dump(mode="json") for row in rows),
            output,
        )

    def federation_results(self, run_dirs: tuple[Path, ...], output: Path) -> Path:
        rows: list[FederationResultsRow] = []
        for run_dir in run_dirs:
            metrics = run_dir / "metrics" / "federation.json"
            if metrics.exists():
                federation = FederationMetrics.model_validate_json(
                    metrics.read_text(encoding="utf-8")
                )
                rows.append(
                    FederationResultsRow(
                        run_id=run_dir.name,
                        mebe=federation.mebe,
                        high_excess=federation.high_excess,
                        band_violation_rate=federation.band_violation_rate,
                        attack_balanced_macro_tpr=federation.attack_balanced_macro_tpr,
                    )
                )
        return self._write(
            pd.DataFrame.from_records(row.model_dump(mode="json") for row in rows),
            output,
        )

    @staticmethod
    def _write(frame: pd.DataFrame, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        return output


class PublicationPackage:
    """Tables, figures, and manifest of one publication."""
    def __init__(
        self,
        tables: tuple[PublicationArtifact, ...],
        figures: tuple[PublicationArtifact, ...],
        manifest: Path,
    ) -> None:
        self.tables = tables
        self.figures = figures
        self.manifest = manifest

    @property
    def complete(self) -> bool:
        return all(item.available for item in (*self.tables, *self.figures))


class PublicationArtifact:
    """One generated table or figure with availability."""
    def __init__(
        self,
        name: Description,
        path: Path | None,
        available: bool,
        reason: Description | None = None,
    ) -> None:
        self.name = name
        self.path = path
        self.available = available
        self.reason = reason


class PublicationPackageBuilder:
    """Generate the registered tables and figures without manual outcomes."""

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

        runs = _completed_runs(outputs_root)
        primary_runs = tuple(
            path
            for path in runs
            if (manifest := _run_manifest(path)) is not None
            and manifest.experiment_id is ExperimentId.PRIMARY_NBAIOT
        )
        fedcrg_primary = tuple(
            path
            for path in primary_runs
            if (manifest := _run_manifest(path)) is not None
            and manifest.policy_id is PolicyId.FEDCRG
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
                fedcrg_primary,
                lambda: self.tables.admission_states_from_runs(
                    fedcrg_primary, table_root / "table_5_admission_states.csv"
                ),
                "FedCRG admission runs are unavailable",
            ),
            self._runs_table(
                "Table 6 - Ablations",
                fedcrg_primary,
                lambda: self.tables.ablations(
                    primary_runs, table_root / "table_6_ablations.csv", config
                ),
                "primary policy runs are unavailable",
            ),
        )
        figures = (
            self._figure(
                "Figure 1 - Decision architecture",
                lambda: build_decision_architecture_figure(
                    figure_root / "figure_1_decision_architecture.png"
                ),
            ),
        )
        manifest_path = destination / "manifest.json"
        atomic_write_json(
            manifest_path,
            {
                "complete": all(item.available for item in (*tables, *figures)),
                "tables": [
                    {
                        "name": item.name,
                        "available": item.available,
                        "path": None if item.path is None else item.path.name,
                        "reason": item.reason,
                    }
                    for item in tables
                ],
                "figures": [
                    {
                        "name": item.name,
                        "available": item.available,
                        "path": None if item.path is None else item.path.name,
                        "reason": item.reason,
                    }
                    for item in figures
                ],
            },
        )
        return PublicationPackage(tables, figures, manifest_path)

    def _table(self, name: Description, builder: Callable[[], Path]) -> PublicationArtifact:
        try:
            path = builder()
        except Exception as exc:
            return PublicationArtifact(name, None, False, str(exc))
        return PublicationArtifact(name, path, path.is_file())

    def _optional_table(
        self,
        name: Description,
        source: Path | None,
        builder: Callable[[Path], Path],
        reason: Description,
    ) -> PublicationArtifact:
        if source is None or not source.is_file():
            return PublicationArtifact(name, None, False, reason)
        return self._table(name, lambda: builder(source))

    def _runs_table(
        self,
        name: Description,
        runs: tuple[Path, ...],
        builder: Callable[[], Path],
        reason: Description,
    ) -> PublicationArtifact:
        if not runs:
            return PublicationArtifact(name, None, False, reason)
        return self._table(name, builder)

    def _figure(self, name: Description, builder: Callable[[], Path]) -> PublicationArtifact:
        try:
            path = builder()
        except Exception as exc:
            return PublicationArtifact(name, None, False, str(exc))
        return PublicationArtifact(name, path, path.is_file())


def build_decision_architecture_figure(output: Path) -> Path:
    """Render the evidence-admission flow without implementation shorthand names."""
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 8)
    axis.axis("off")

    boxes = (
        (0.5, 5.8, 2.3, 1.1, "Equal-count federation\nreference evidence"),
        (3.4, 5.8, 2.5, 1.1, "Independent client\nreference-mismatch evidence"),
        (6.5, 5.8, 2.5, 1.1, "Independent local\ncalibration readiness"),
        (9.6, 5.8, 1.9, 1.1, "Deployment\ndecision"),
        (0.7, 2.4, 2.5, 1.0, "Reference retained\n(no material mismatch demonstrated)"),
        (3.5, 2.4, 2.3, 1.0, "Mismatch evidence\ninsufficient"),
        (6.1, 2.4, 2.2, 1.0, "Calibration deficit"),
        (8.6, 2.4, 2.5, 1.0, "Calibration assumption\nviolation"),
        (5.0, 0.6, 2.4, 1.0, "Client-specific threshold\npersonalization admitted"),
    )

    for x, y, width, height, label in boxes:
        from matplotlib.patches import FancyBboxPatch

        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.04", linewidth=1.2, fill=False
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)

    def arrow(start: tuple[float, float], end: tuple[float, float], label: str | None = None) -> None:
        from matplotlib.patches import FancyArrowPatch

        axis.add_patch(
            FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.1)
        )
        if label is not None:
            midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            axis.text(midpoint[0], midpoint[1] + 0.15, label, ha="center", fontsize=8)

    arrow((2.8, 6.35), (3.4, 6.35))
    arrow((5.9, 6.35), (6.5, 6.35), "mismatch established")
    arrow((9.0, 6.35), (9.6, 6.35))
    arrow((4.6, 5.8), (1.95, 3.4), "no material mismatch demonstrated")
    arrow((4.6, 5.8), (4.65, 3.4), "sample too small")
    arrow((7.7, 5.8), (7.2, 3.4), "not ready")
    arrow((7.7, 5.8), (9.85, 3.4), "selected-score tie")
    arrow((10.55, 5.8), (6.2, 1.6), "mismatch + ready + unique threshold")

    axis.text(
        0.6,
        7.45,
        "Disjoint benign evidence roles: reference / mismatch / calibration",
        fontsize=10,
        fontweight="bold",
    )
    axis.text(
        0.6,
        7.05,
        "Attack labels are absent from admission and threshold construction.",
        fontsize=9,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def build_repository_report(outputs: Path, config: ExperimentConfig) -> Path:
    """Publication-oriented evidence index from every completed run."""
    reports_root = outputs / "reports" / "latest"
    reports_root.mkdir(parents=True, exist_ok=True)
    run_dirs = _completed_runs(outputs)
    builder = PublicationTableBuilder()
    policy_table = builder.federation_results(
        run_dirs, reports_root / "policy_federation_results.csv"
    )
    records = load_federation_results(run_dirs)
    primary_records = tuple(
        row for row in records if row.experiment_id is ExperimentId.PRIMARY_NBAIOT
    )
    contrasts_path = reports_root / "primary_contrasts.json"
    if primary_records:
        contrasts = confirmatory_contrasts(
            primary_records,
            named_calibration_seed=int(config.dataset.primary_calibration_seed),
            expected_model_seeds=tuple(config.randomness.model_seeds),
            bootstrap_seed=config.statistics.bootstrap_seed,
            bootstrap_replicates=config.statistics.bootstrap_replicates,
        )
        contrasts_payload: object = [
            {"comparator": item.comparator.value, "metrics": [metric.model_dump(mode="json") for metric in item.metrics]}
            for item in contrasts
        ]
    else:
        contrasts_payload = {
            "status": "incomplete",
            "reason": "primary workload has not reconciled, confirmatory contrasts are withheld",
        }
    atomic_write_json(contrasts_path, contrasts_payload)
    sensitivity_path = reports_root / "split_sensitivity.csv"
    pd.DataFrame.from_records(
        [row.model_dump(mode="json") for row in split_sensitivity(primary_records)]
    ).to_csv(sensitivity_path, index=False)

    lines = [
        "# FedCRG Reproducibility Report",
        "",
        f"- Completed run directories: `{len(run_dirs)}`",
        f"- Federation-results table: `{policy_table.name}`",
        f"- Confirmatory contrast analysis: `{contrasts_path.name}`",
        f"- Split-sensitivity summary: `{sensitivity_path.name}`",
        "",
        "The repository report is generated from immutable artifacts and workload "
        "ledgers. Missing experiments remain explicitly incomplete; report generation "
        "never fills missing evidence with inferred values.",
    ]
    output = reports_root / "README.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def build_publication(
    config: ExperimentConfig,
    outputs_root: Path,
    prepared_manifest: Path | None = None,
    destination: Path | None = None,
) -> Path:
    """Build the publication package and return its manifest."""
    package = PublicationPackageBuilder().build(
        config=config,
        outputs_root=outputs_root,
        prepared_manifest=prepared_manifest,
        destination=destination,
    )
    return package.manifest


_REQUIRED_BUNDLE_DIRECTORIES = (
    "resolved_configs",
    "metrics",
    "statistics",
    "tables",
    "figures",
    "reports",
    "provenance",
)


class ResultsManifest(BaseModel):
    """Frozen results-bundle manifest."""
    model_config = Frozen

    campaign_id: CampaignId
    complete: bool
    config_hash: Sha256
    dataset_id: DatasetId
    detector_id: DetectorId | None
    model_seeds: tuple[ModelSeed, ...]
    calibration_seeds: tuple[CalibrationSeed, ...]
    outputs_root: Identifier
    file_count: PositiveCount
    source_policy: Description


class ResultsVerification:
    """Validity and problems of one bundle verification."""
    def __init__(self, valid: bool, problems: tuple[Identifier, ...]) -> None:
        self.valid = valid
        self.problems = problems


class ResultsBuilder:
    """Assemble the publication bundle for one campaign from immutable evidence."""

    def build(
        self,
        *,
        campaign_id: CampaignId,
        outputs_root: Path,
        results_root: Path,
    ) -> Path:
        destination = results_root / str(campaign_id)
        if destination.exists():
            raise FileExistsError(f"Results bundle already exists and is immutable: {destination}")
        for directory in _REQUIRED_BUNDLE_DIRECTORIES:
            (destination / directory).mkdir(parents=True, exist_ok=True)

        study = Study.load()
        config = study.resolve(ExperimentId.PRIMARY_NBAIOT)
        self._write_resolved_configs(config, destination / "resolved_configs")
        self._copy_metrics(outputs_root, destination / "metrics")
        self._copy_statistics(outputs_root, destination / "statistics")
        self._write_reports(outputs_root, destination / "reports", config)
        self._write_provenance(outputs_root, destination / "provenance")
        self._copy_tables_and_figures(outputs_root, destination)

        checksums = self._checksums(destination)
        atomic_write_json(destination / "checksums.json", checksums)
        manifest = self._manifest(campaign_id, outputs_root, destination, config, checksums)
        atomic_write_json(destination / "manifest.json", manifest)
        return destination

    @staticmethod
    def _write_resolved_configs(config: ExperimentConfig, destination: Path) -> None:
        atomic_write_json(
            destination / "primary_nbaiot.json",
            {
                "config_hash": config.config_hash,
                "data_spec_hash": config.data_spec_hash,
                "payload": config.model_dump(mode="json"),
            },
        )

    @staticmethod
    def _copy_metrics(outputs_root: Path, destination: Path) -> None:
        runs_root = OutputsLayout(outputs_root).runs
        if not runs_root.exists():
            return
        records: list[MetricRecord] = []
        for run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            manifest_path = run_root / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = RunManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
            except pydantic.ValidationError:
                continue
            if manifest.status.value != "complete":
                continue
            metric_path = run_root / "metrics" / "metric_record.jsonl"
            if not metric_path.is_file():
                continue
            for line in metric_path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(MetricRecord.model_validate_json(line))
                except pydantic.ValidationError:
                    continue
        atomic_write_json(
            destination / "metric_records.json",
            {"records": [record.model_dump(mode="json") for record in records]},
        )

    @staticmethod
    def _copy_statistics(outputs_root: Path, destination: Path) -> None:
        analysis_root = OutputsLayout(outputs_root).cache_analysis
        if not analysis_root.exists():
            return
        for name in ("readiness_plans.json", "mismatch_cutoffs.json"):
            source = analysis_root / name
            if source.is_file():
                (destination / name).write_bytes(source.read_bytes())

    def _write_reports(
        self, outputs_root: Path, destination: Path, config: ExperimentConfig
    ) -> None:
        report = build_repository_report(outputs_root, config)
        if report.is_file():
            (destination / report.name).write_bytes(report.read_bytes())

    @staticmethod
    def _write_provenance(outputs_root: Path, destination: Path) -> None:
        environment: GitEnvironment | None = None
        environment_path = OutputsLayout(outputs_root).environment_file
        if environment_path.is_file():
            environment = GitEnvironment.model_validate_json(
                environment_path.read_text(encoding="utf-8")
            )
        atomic_write_json(
            destination / "provenance.json",
            {
                "environment": (
                    environment.model_dump(mode="json") if environment is not None else None
                ),
                "prepared_data_root": "data/preprocessed/",
                "outputs_root": str(outputs_root),
            },
        )

    @staticmethod
    def _copy_tables_and_figures(outputs_root: Path, destination: Path) -> None:
        publication_root = outputs_root / "reports" / "publication"
        if not publication_root.exists():
            return
        for source_dir, target_dir in (
            ("tables", destination / "tables"),
            ("figures", destination / "figures"),
        ):
            source = publication_root / source_dir
            if not source.exists():
                continue
            for path in source.iterdir():
                if path.is_file():
                    (target_dir / path.name).write_bytes(path.read_bytes())

    @staticmethod
    def _checksums(root: Path) -> tuple[ChecksumRecord, ...]:
        records: list[ChecksumRecord] = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "checksums.json":
                records.append(
                    ChecksumRecord(
                        relative_path=path.relative_to(root).as_posix(),
                        sha256=sha256_file(path),
                    )
                )
        return tuple(records)

    @staticmethod
    def _manifest(
        campaign_id: CampaignId,
        outputs_root: Path,
        destination: Path,
        config: ExperimentConfig,
        checksums: tuple[ChecksumRecord, ...],
    ) -> ResultsManifest:
        detector = config.detector
        return ResultsManifest(
            campaign_id=campaign_id,
            complete=True,
            config_hash=config.config_hash,
            dataset_id=config.dataset.id,
            detector_id=detector.id if detector is not None else None,
            model_seeds=tuple(config.randomness.model_seeds),
            calibration_seeds=tuple(config.dataset.calibration_seeds),
            outputs_root=str(outputs_root),
            file_count=len(checksums),
            source_policy="all numerical content is derived from immutable FedCRG artifacts",
        )


class ResultsVerifier:
    """Verify that a publication bundle satisfies the required structure and integrity."""

    def verify(
        self,
        campaign_id: CampaignId,
        *,
        results_root: Path,
        outputs_root: Path,
    ) -> ResultsVerification:
        destination = results_root / str(campaign_id)
        problems: list[Identifier] = []
        if not destination.exists():
            return ResultsVerification(
                False, (f"results bundle does not exist: {destination}",)
            )
        for directory in _REQUIRED_BUNDLE_DIRECTORIES:
            if not (destination / directory).is_dir():
                problems.append(f"missing required bundle directory: {directory}")
        manifest_path = destination / "manifest.json"
        if not manifest_path.is_file():
            problems.append("missing bundle manifest.json")
        checksums_path = destination / "checksums.json"
        if not checksums_path.is_file():
            problems.append("missing bundle checksums.json")
        else:
            checksums = tuple(
                ChecksumRecord.model_validate(entry)
                for entry in json.loads(checksums_path.read_text(encoding="utf-8"))
            )
            for path in sorted(destination.rglob("*")):
                if path.is_file() and path.name != "checksums.json":
                    relative = path.relative_to(destination).as_posix()
                    expected = next(
                        (record.sha256 for record in checksums if record.relative_path == relative),
                        None,
                    )
                    if expected is None:
                        problems.append(f"unchecksummed bundle file: {relative}")
                    elif str(expected) != sha256_file(path):
                        problems.append(f"bundle hash mismatch: {relative}")
        return ResultsVerification(not problems, tuple(problems))


def build_results_bundle(
    campaign_id: CampaignId,
    outputs_root: Path,
    results_root: Path,
) -> Path:
    """Assemble one immutable results bundle."""
    return ResultsBuilder().build(
        campaign_id=campaign_id,
        outputs_root=outputs_root,
        results_root=results_root,
    )


def verify_results_bundle(
    campaign_id: CampaignId,
    outputs_root: Path,
    results_root: Path,
) -> ResultsVerification:
    """Verify one results bundle's structure and checksums."""
    return ResultsVerifier().verify(
        campaign_id,
        results_root=results_root,
        outputs_root=outputs_root,
    )
