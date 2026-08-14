"""Reproducible reports and publication bundles built only from immutable evidence.

Run reports, the repository report, publication tables and figures, and the
results bundle are projections of frozen run artifacts: no detector is loaded,
no model is retrained, and missing evidence remains explicitly incomplete.
"""

from __future__ import annotations

import json

import pydantic
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import TypeAdapter

from fedcrg.config import AxisValue, ExperimentConfig, Study
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
    atomic_write_json,
)
from fedcrg.hashing import sha256_file
from fedcrg.paths import (
    LayoutDirectory,
    OutputsLayout,
    PublicationLayout,
    ResultsBundleLayout,
    RunLayout,
    campaign_results_root,
)
from fedcrg.experiments.analyses import (
    confirmatory_contrasts,
    load_federation_results,
    split_sensitivity,
)
from fedcrg.thresholding.metrics import FederationMetrics
from fedcrg.types import (
    Assurance,
    CalibrationSeed,
    CampaignId,
    ClientId,
    ConfidenceLevel,
    DataRole,
    DecisionSource,
    DecisionState,
    Description,
    DetectorId,
    DatasetId,
    ExperimentAxisId,
    ExperimentId,
    ExperimentStatus,
    ExperimentType,
    FailureCode,
    Fpr,
    Fraction,
    Identifier,
    Metric,
    ModelSeed,
    NonNegativeCount,
    OperatingBand,
    PathString,
    PolicyId,
    SampleCount,
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

_FPR_ADAPTER = TypeAdapter(Fpr)


class PublicationTableFilename(StrEnum):
    """Reserved manuscript-table filenames under a publication's tables/ root."""

    PROTOCOL_CONSTANTS = "table_1_protocol_constants.csv"
    DATASET_INVENTORY = "table_2_dataset_inventory.csv"
    PRIMARY_POLICY_RESULTS = "table_3_primary_policy_results.csv"
    ADMISSION_STATES = "table_4_admission_states.csv"
    ABLATIONS = "table_5_ablations.csv"
    EXTERNAL_REPLICATION = "table_8_external_replication.csv"


class RepositoryReportFilename(StrEnum):
    """Reserved filenames under the repository report's reports/latest/ root."""

    POLICY_FEDERATION_RESULTS = "policy_federation_results.csv"
    PRIMARY_CONTRASTS = "primary_contrasts.json"
    SPLIT_SENSITIVITY = "split_sensitivity.csv"
    README = "README.md"


_RUN_SUMMARY_FILENAME = "summary.md"


def _run_manifest(run_dir: Path) -> RunManifest | None:
    manifest_path = RunLayout(run_dir).manifest
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
        manifest = _run_manifest(path)
        if manifest is not None and manifest.status is ExperimentStatus.COMPLETE:
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
    output = layout.reports / _RUN_SUMMARY_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _definition_for(manifest: RunManifest):
    """Resolve the catalogue definition recorded in the run manifest."""
    return Study.load().catalogue.spec(manifest.experiment_id)


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
                    records.append(ThresholdRecord.model_validate_json(line))
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
            ProtocolConstantRow(constant="readiness_assurance", value=protocol.readiness_assurance),
            ProtocolConstantRow(constant="mismatch_confidence", value=protocol.mismatch_confidence),
        ]
        training = config.training
        if training is not None:
            rows.extend(
                [
                    ProtocolConstantRow(constant="rounds", value=training.rounds),
                    ProtocolConstantRow(constant="local_epochs", value=training.local_epochs),
                    ProtocolConstantRow(constant="batch_size", value=training.batch_size),
                    ProtocolConstantRow(
                        constant="learning_rate_initial",
                        value=training.learning_rate_initial,
                    ),
                    ProtocolConstantRow(
                        constant="learning_rate_final",
                        value=training.learning_rate_final,
                    ),
                    ProtocolConstantRow(constant="client_fraction", value=training.client_fraction),
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
            metrics = RunLayout(run_dir).federation_metrics
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


@dataclass(frozen=True, slots=True)
class PublicationPackage:
    """Tables, figures, and manifest of one publication."""

    tables: tuple[PublicationArtifact, ...]
    figures: tuple[PublicationArtifact, ...]
    manifest: Path

    @property
    def complete(self) -> bool:
        return all(item.available for item in (*self.tables, *self.figures))


@dataclass(frozen=True, slots=True)
class PublicationArtifact:
    """One generated table or figure with availability."""

    name: Description
    path: Path | None
    available: bool
    reason: Description | None = None


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
        destination = destination or OutputsLayout(outputs_root).publication.root
        publication = PublicationLayout(destination)
        table_root = publication.tables
        figure_root = publication.figures
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
                "Table 1 - Protocol constants",
                lambda: self.tables.protocol_constants(
                    config, table_root / PublicationTableFilename.PROTOCOL_CONSTANTS.value
                ),
            ),
            self._optional_table(
                "Table 2 - Dataset inventory",
                prepared_manifest,
                lambda path: self.tables.dataset_inventory(
                    path, table_root / PublicationTableFilename.DATASET_INVENTORY.value
                ),
                "prepared dataset manifest is unavailable",
            ),
            self._runs_table(
                "Table 3 - Primary policy results",
                primary_runs,
                lambda: self.tables.primary_policy_results(
                    primary_runs, table_root / PublicationTableFilename.PRIMARY_POLICY_RESULTS.value
                ),
                "primary policy runs are unavailable",
            ),
            self._runs_table(
                "Table 4 - Admission states",
                fedcrg_primary,
                lambda: self.tables.admission_states_from_runs(
                    fedcrg_primary, table_root / PublicationTableFilename.ADMISSION_STATES.value
                ),
                "FedCRG admission runs are unavailable",
            ),
            self._runs_table(
                "Table 5 - Ablations",
                fedcrg_primary,
                lambda: self.tables.ablations(
                    primary_runs, table_root / PublicationTableFilename.ABLATIONS.value, config
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
            self._figure(
                "Figure 2 - Finite-sample readiness frontier",
                lambda: build_readiness_frontier_from_catalogue(
                    figure_root / "figure_2_readiness_frontier.png"
                ),
            ),
            self._figure(
                "Figure 3 - Reference-mismatch evidence/power map",
                lambda: build_mismatch_power_map_from_catalogue(
                    figure_root / "figure_3_mismatch_power_map.png"
                ),
            ),
            self._figure(
                "Figure 4 - Per-client operating points",
                lambda: build_per_client_operating_points_figure(
                    figure_root / "figure_4_per_client_operating_points.png", destination
                ),
            ),
            self._figure(
                "Figure 5 - Reliability-utility frontier",
                lambda: build_reliability_utility_frontier_figure(
                    figure_root / "figure_5_reliability_utility_frontier.png", destination
                ),
            ),
            self._figure(
                "Figure 6 - Calibration-size phase transition",
                lambda: build_phase_transition_from_catalogue(
                    figure_root / "figure_6_phase_transition.png"
                ),
            ),
            self._figure(
                "Figure 7 - Assumption stress",
                lambda: build_assumption_stress_figure(
                    figure_root / "figure_7_assumption_stress.png", destination
                ),
            ),
            self._figure(
                "Figure 8 - External replication",
                lambda: build_external_replication_figure(
                    figure_root / "figure_8_external_replication.png", destination
                ),
            ),
        )
        manifest_path = publication.manifest
        atomic_write_json(
            manifest_path,
            {
                ExperimentStatus.COMPLETE.value: all(
                    item.available for item in (*tables, *figures)
                ),
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

    def arrow(
        start: tuple[float, float], end: tuple[float, float], label: str | None = None
    ) -> None:
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


def build_readiness_frontier_figure(
    output: Path,
    sample_counts: tuple[SampleCount, ...],
    band: OperatingBand,
    assurance: Assurance,
) -> Path:
    """Render the finite-sample readiness frontier (n_C vs exact in-band probability)."""
    from fedcrg.thresholding.readiness import ReadinessPlanBuilder

    probabilities = [
        ReadinessPlanBuilder().build(sample_count, band, assurance).coverage_probability
        for sample_count in sample_counts
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(sample_counts, probabilities, marker="o", linewidth=1.4)
    axis.axhline(assurance, color="crimson", linestyle="--", linewidth=1.0)
    minimum = next(
        (
            sample_count
            for sample_count, p in zip(sample_counts, probabilities, strict=True)
            if p >= assurance
        ),
        None,
    )
    if minimum is not None:
        axis.axvline(minimum, color="slategray", linestyle=":", linewidth=1.0)
        axis.annotate(
            f"minimum ready n_C={minimum}",
            xy=(minimum, assurance),
            xytext=(minimum * 0.8, assurance + 0.02),
            fontsize=8,
        )
    axis.set_xlabel("local calibration sample count n_C")
    axis.set_ylabel("maximum exact in-band probability")
    axis.set_title("Finite-sample readiness frontier")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def build_mismatch_power_map_figure(
    output: Path,
    sample_counts: tuple[SampleCount, ...],
    true_fprs: tuple[Fpr, ...],
    band: OperatingBand,
    confidence: ConfidenceLevel,
) -> Path:
    """Render the reference-mismatch evidence/power map (n_G x true reference FPR)."""
    from scipy.stats import binom

    from fedcrg.thresholding.readiness import clopper_pearson_interval
    from fedcrg.types import BinomialCounts

    rows: list[list[Fraction]] = []
    for sample_count in sample_counts:
        powers: list[Fraction] = []
        for true_fpr in true_fprs:
            power = 0.0
            for exceedances in range(sample_count + 1):
                interval = clopper_pearson_interval(
                    BinomialCounts(exceedances, sample_count), confidence
                )
                if interval.upper < band.lower or interval.lower > band.upper:
                    power += float(binom.pmf(exceedances, sample_count, true_fpr))
            powers.append(power)
        rows.append(powers)
    grid = np.asarray(rows, dtype=np.float64)
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=(
            float(min(true_fprs)),
            float(max(true_fprs)),
            float(min(sample_counts)),
            float(max(sample_counts)),
        ),
    )
    figure.colorbar(image, ax=axis, label="mismatch-declaration probability")
    axis.set_xlabel("true reference FPR")
    axis.set_ylabel("gate sample count n_G")
    axis.set_title("Reference-mismatch evidence/power map")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def build_phase_transition_figure(
    output: Path,
    calibration_counts: tuple[SampleCount, ...],
    gate_counts: tuple[SampleCount, ...],
    band: OperatingBand,
    assurance: Assurance,
    confidence: ConfidenceLevel,
) -> Path:
    """Render the calibration-size phase transition (readiness and mismatch evidence)."""
    from fedcrg.thresholding.readiness import (
        ReadinessPlanBuilder,
        minimum_bidirectional_sample_count,
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    probabilities = [
        ReadinessPlanBuilder().build(sample_count, band, assurance).coverage_probability
        for sample_count in calibration_counts
    ]
    axes[0].plot(calibration_counts, probabilities, marker="o", linewidth=1.4)
    axes[0].axhline(assurance, color="crimson", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("n_C")
    axes[0].set_ylabel("in-band probability")
    axes[0].set_title("Local readiness evidence")
    minimum = minimum_bidirectional_sample_count(band.lower, confidence)
    axes[1].set_xlim(min(gate_counts), max(gate_counts))
    axes[1].axvline(minimum, color="crimson", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("n_G")
    axes[1].set_ylabel("bidirectional minimum")
    axes[1].set_title("Mismatch evidence minimum")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def _require_bundle_table(results_root: Path, filename: Identifier) -> pd.DataFrame:
    """Load one publication table; results-driven figures never fabricate data."""
    path = ResultsBundleLayout(results_root).tables / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Results-driven figure requires the {filename} table, run the campaign first ({path})"
        )
    return pd.read_csv(path)


def _band_guide_lines() -> tuple[Fpr, Fpr, Fpr]:
    """The locked operating band guide lines from the study protocol."""
    protocol = Study.load().study_config.protocol
    return protocol.band.lower, protocol.alpha, protocol.band.upper


def build_per_client_operating_points_figure(output: Path, results_root: Path) -> Path:
    """Render per-client FPR operating points against the locked band lines."""
    frame = _require_bundle_table(results_root, PublicationTableFilename.PRIMARY_POLICY_RESULTS.value)
    if "client_id" not in frame.columns or "fpr" not in frame.columns:
        raise ValueError("primary policy results table lacks client_id/fpr columns")
    figure, axis = plt.subplots(figsize=(9, 5))
    for policy, group in frame.groupby("policy_id", sort=True):
        axis.scatter(group["client_id"], group["fpr"], label=policy, s=18)
    for level in _band_guide_lines():
        axis.axhline(level, color="gray", linestyle=":", linewidth=0.8)
    axis.set_xlabel("client id")
    axis.set_ylabel("final-test FPR")
    axis.set_title("Per-client operating points")
    axis.legend(fontsize=8, ncol=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def build_reliability_utility_frontier_figure(output: Path, results_root: Path) -> Path:
    """Render the reliability-utility frontier (MEBE vs ABMacroTPR per policy)."""
    frame = _require_bundle_table(results_root, PublicationTableFilename.PRIMARY_POLICY_RESULTS.value)
    required = {"policy_id", "mebe", "attack_balanced_macro_tpr"}
    if not required.issubset(frame.columns):
        raise ValueError("primary policy results table lacks MEBE/ABMacroTPR columns")
    summary = frame.groupby("policy_id", sort=True)[["mebe", "attack_balanced_macro_tpr"]].mean()
    summary = pd.DataFrame(summary).reset_index()
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(summary["mebe"], summary["attack_balanced_macro_tpr"], s=42)
    for _, row in summary.iterrows():
        policy_label = str(row["policy_id"])
        axis.annotate(policy_label, (row["mebe"], row["attack_balanced_macro_tpr"]), fontsize=8)
    axis.set_xlabel("mean excess band error (MEBE)")
    axis.set_ylabel("attack-balanced macro TPR")
    axis.set_title("Reliability-utility frontier")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def build_assumption_stress_figure(output: Path, results_root: Path) -> Path:
    """Render assumption-stress coverage from the campaign statistics bundle."""
    statistics_root = ResultsBundleLayout(results_root).statistics
    if not statistics_root.is_dir():
        raise FileNotFoundError(
            f"Assumption-stress figure requires campaign statistics, run the synthetic stresses first ({statistics_root})"
        )
    frames: list[pd.DataFrame] = []
    for path in sorted(statistics_root.glob("*.json")):
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "coverage" in payload:
            frames.append(pd.DataFrame([{"cell": path.stem, "coverage": payload["coverage"]}]))
    if not frames:
        raise FileNotFoundError(f"No coverage statistics found under {statistics_root}")
    combined = pd.concat(frames, ignore_index=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(combined["cell"], combined["coverage"])
    axis.set_xlabel("stress cell")
    axis.set_ylabel("realized in-band coverage")
    axis.set_title("Assumption stress")
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def build_external_replication_figure(output: Path, results_root: Path) -> Path:
    """Render external-replication FPR operating points from the DIAD bundle."""
    frame = _require_bundle_table(results_root, PublicationTableFilename.EXTERNAL_REPLICATION.value)
    if "client_id" not in frame.columns or "fpr" not in frame.columns:
        raise ValueError("external replication table lacks client_id/fpr columns")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(range(len(frame)), frame["fpr"], s=18)
    for level in _band_guide_lines():
        axis.axhline(level, color="gray", linestyle=":", linewidth=0.8)
    axis.set_xlabel("eligible DIAD client")
    axis.set_ylabel("final-test FPR")
    axis.set_title("External replication")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def _catalogue_axis_values(
    spec_id: ExperimentId, axis_id: ExperimentAxisId
) -> tuple[AxisValue, ...]:
    """Resolve one locked experiment-grid axis from the catalogue."""
    return tuple(Study.load().catalogue.spec(spec_id).axis(axis_id).values)


def build_readiness_frontier_from_catalogue(output: Path) -> Path:
    """Render the readiness frontier using the locked readiness sweep."""
    counts = tuple(
        int(value)
        for value in _catalogue_axis_values(
            ExperimentId.READINESS_SAMPLE_SIZE, ExperimentAxisId.CALIBRATION_N
        )
    )
    protocol = Study.load().study_config.protocol
    return build_readiness_frontier_figure(
        output, counts, protocol.band, protocol.readiness_assurance
    )


def build_mismatch_power_map_from_catalogue(output: Path) -> Path:
    """Render the mismatch power map using the locked evidence-budget grids."""
    sample_counts = tuple(
        int(value)
        for value in _catalogue_axis_values(
            ExperimentId.MISMATCH_SAMPLE_SIZE, ExperimentAxisId.MISMATCH_N
        )
    )
    true_fprs = tuple(
        _FPR_ADAPTER.validate_python(value)
        for value in _catalogue_axis_values(ExperimentId.MISMATCH_POWER, ExperimentAxisId.TRUE_FPR)
    )
    protocol = Study.load().study_config.protocol
    return build_mismatch_power_map_figure(
        output, sample_counts, true_fprs, protocol.band, protocol.mismatch_confidence
    )


def build_phase_transition_from_catalogue(output: Path) -> Path:
    """Render the calibration-size phase transition from the locked sweeps."""
    calibration_counts = tuple(
        int(value)
        for value in _catalogue_axis_values(
            ExperimentId.READINESS_SAMPLE_SIZE, ExperimentAxisId.CALIBRATION_N
        )
    )
    gate_counts = tuple(
        int(value)
        for value in _catalogue_axis_values(
            ExperimentId.MISMATCH_SAMPLE_SIZE, ExperimentAxisId.MISMATCH_N
        )
    )
    protocol = Study.load().study_config.protocol
    return build_phase_transition_figure(
        output,
        calibration_counts,
        gate_counts,
        protocol.band,
        protocol.readiness_assurance,
        protocol.mismatch_confidence,
    )


def build_repository_report(outputs_root: Path, config: ExperimentConfig) -> Path:
    """Publication-oriented evidence index from every completed run."""
    reports_root = OutputsLayout(outputs_root).reports / LayoutDirectory.LATEST.value
    reports_root.mkdir(parents=True, exist_ok=True)
    run_dirs = _completed_runs(outputs_root)
    builder = PublicationTableBuilder()
    policy_table = builder.federation_results(
        run_dirs, reports_root / RepositoryReportFilename.POLICY_FEDERATION_RESULTS.value
    )
    records = load_federation_results(run_dirs)
    primary_records = tuple(
        row for row in records if row.experiment_id is ExperimentId.PRIMARY_NBAIOT
    )
    contrasts_path = reports_root / RepositoryReportFilename.PRIMARY_CONTRASTS.value
    if primary_records:
        contrasts = confirmatory_contrasts(
            primary_records,
            named_calibration_seed=int(config.dataset.primary_calibration_seed),
            expected_model_seeds=tuple(config.randomness.model_seeds),
            bootstrap_seed=config.statistics.bootstrap_seed,
            bootstrap_replicates=config.statistics.bootstrap_replicates,
        )
        contrasts_payload = [
            {
                "comparator": item.comparator.value,
                "metrics": [metric.model_dump(mode="json") for metric in item.metrics],
            }
            for item in contrasts
        ]
    else:
        contrasts_payload = {
            "status": "incomplete",
            "reason": "primary workload has not reconciled, confirmatory contrasts are withheld",
        }
    atomic_write_json(contrasts_path, contrasts_payload)
    sensitivity_path = reports_root / RepositoryReportFilename.SPLIT_SENSITIVITY.value
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
    output = reports_root / RepositoryReportFilename.README.value
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
    outputs_root: PathString
    file_count: PositiveCount
    source_policy: Description


@dataclass(frozen=True, slots=True)
class ResultsVerification:
    """Validity and problems of one bundle verification."""

    valid: bool
    problems: tuple[Identifier, ...]


class ResultsBuilder:
    """Assemble the publication bundle for one campaign from immutable evidence."""

    def build(
        self,
        *,
        campaign_id: CampaignId,
        outputs_root: Path,
        results_root: Path,
    ) -> Path:
        destination = campaign_results_root(results_root, campaign_id)
        layout = ResultsBundleLayout(destination)
        if destination.exists():
            raise FileExistsError(f"Results bundle already exists and is immutable: {destination}")
        for directory in layout.required_directories:
            directory.mkdir(parents=True, exist_ok=True)

        study = Study.load()
        config = study.resolve(ExperimentId.PRIMARY_NBAIOT)
        self._write_resolved_configs(config, layout)
        self._copy_metrics(outputs_root, layout)
        self._copy_statistics(outputs_root, layout)
        self._write_reports(outputs_root, config, layout)
        self._write_provenance(outputs_root, config, layout)
        self._copy_tables_and_figures(outputs_root, layout)

        # The manifest itself must be covered by the checksum ledger, so it is
        # written before the ledger is computed (checksums.json is the only
        # file excluded from its own ledger).
        complete = self._evidence_complete(study, outputs_root)
        checksums = self._checksums(layout)
        manifest = self._manifest(
            campaign_id,
            outputs_root,
            config,
            checksums,
            complete,
            file_count=len(checksums) + 1,
        )
        atomic_write_json(layout.manifest, manifest)
        checksums = self._checksums(layout)
        atomic_write_json(layout.checksums, checksums)
        return destination

    @staticmethod
    def _evidence_complete(study: Study, outputs_root: Path) -> bool:
        """True when every registered real-data experiment cell has a completed run.

        The bundle manifest must never claim completeness for missing evidence:
        a partial programme stays explicitly incomplete until every
        (model seed, calibration seed, policy) cell of every real-data
        experiment exists as a completed run directory.
        """
        runs = _completed_runs(outputs_root)
        for spec in study.catalogue.all():
            if spec.category is ExperimentType.SYNTHETIC:
                continue
            if spec.id is ExperimentId.COMPUTATIONAL_BENCHMARK:
                continue
            config = study.resolve(spec.id)
            expected = (
                len(config.randomness.model_seeds)
                * len(config.dataset.calibration_seeds)
                * len(config.policies)
            )
            if expected == 0:
                continue
            observed = sum(
                1
                for path in runs
                if (manifest := _run_manifest(path)) is not None
                and manifest.experiment_id is spec.id
            )
            if observed != expected:
                return False
        return True

    @staticmethod
    def _write_resolved_configs(config: ExperimentConfig, layout: ResultsBundleLayout) -> None:
        atomic_write_json(
            layout.primary_nbaiot_config,
            {
                "config_hash": config.config_hash,
                "data_spec_hash": config.data_spec_hash,
                "payload": config.model_dump(mode="json"),
            },
        )

    @staticmethod
    def _copy_metrics(outputs_root: Path, layout: ResultsBundleLayout) -> None:
        runs_root = OutputsLayout(outputs_root).runs
        if not runs_root.exists():
            return
        records: list[MetricRecord] = []
        for run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            manifest = _run_manifest(run_root)
            if manifest is None or manifest.status is not ExperimentStatus.COMPLETE:
                continue
            run_layout = RunLayout(run_root)
            metric_path = run_layout.metric_records
            if not metric_path.is_file():
                continue
            for line in metric_path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(MetricRecord.model_validate_json(line))
                except pydantic.ValidationError:
                    continue
        atomic_write_json(
            layout.metric_records,
            {"records": [record.model_dump(mode="json") for record in records]},
        )

    @staticmethod
    def _copy_statistics(outputs_root: Path, layout: ResultsBundleLayout) -> None:
        outputs_layout = OutputsLayout(outputs_root)
        for source, target in (
            (outputs_layout.readiness_plans_file, layout.readiness_plans),
            (outputs_layout.mismatch_cutoffs_file, layout.mismatch_cutoffs),
        ):
            if source.is_file():
                target.write_bytes(source.read_bytes())

    def _write_reports(
        self, outputs_root: Path, config: ExperimentConfig, layout: ResultsBundleLayout
    ) -> None:
        report = build_repository_report(outputs_root, config)
        if report.is_file():
            (layout.reports / report.name).write_bytes(report.read_bytes())

    @staticmethod
    def _write_provenance(
        outputs_root: Path, config: ExperimentConfig, layout: ResultsBundleLayout
    ) -> None:
        environment: GitEnvironment | None = None
        environment_path = OutputsLayout(outputs_root).environment_file
        if environment_path.is_file():
            environment = GitEnvironment.model_validate_json(
                environment_path.read_text(encoding="utf-8")
            )
        atomic_write_json(
            layout.provenance_json,
            {
                "environment": (
                    environment.model_dump(mode="json") if environment is not None else None
                ),
                "prepared_data_root": config.preprocessed_root.as_posix(),
                "outputs_root": outputs_root.as_posix(),
            },
        )

    @staticmethod
    def _copy_tables_and_figures(outputs_root: Path, layout: ResultsBundleLayout) -> None:
        publication = OutputsLayout(outputs_root).publication
        if not publication.root.exists():
            return
        for source, target_dir in (
            (publication.tables, layout.tables),
            (publication.figures, layout.figures),
        ):
            if not source.exists():
                continue
            for path in source.iterdir():
                if path.is_file():
                    (target_dir / path.name).write_bytes(path.read_bytes())

    @staticmethod
    def _checksums(layout: ResultsBundleLayout) -> tuple[ChecksumRecord, ...]:
        root = layout.root
        records: list[ChecksumRecord] = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != layout.checksums.name:
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
        config: ExperimentConfig,
        checksums: tuple[ChecksumRecord, ...],
        complete: bool,
        file_count: PositiveCount | None = None,
    ) -> ResultsManifest:
        detector = config.detector
        return ResultsManifest(
            campaign_id=campaign_id,
            complete=complete,
            config_hash=config.config_hash,
            dataset_id=config.dataset.id,
            detector_id=detector.id if detector is not None else None,
            model_seeds=tuple(config.randomness.model_seeds),
            calibration_seeds=tuple(config.dataset.calibration_seeds),
            outputs_root=outputs_root.as_posix(),
            file_count=file_count if file_count is not None else len(checksums),
            source_policy="all numerical content is derived from immutable FedCRG artifacts",
        )


class ResultsVerifier:
    """Verify that a publication bundle satisfies the required structure and integrity."""

    def verify(
        self,
        campaign_id: CampaignId,
        *,
        results_root: Path,
    ) -> ResultsVerification:
        destination = campaign_results_root(results_root, campaign_id)
        layout = ResultsBundleLayout(destination)
        problems: list[Identifier] = []
        if not destination.exists():
            return ResultsVerification(False, (f"results bundle does not exist: {destination}",))
        for directory in layout.required_directories:
            if not directory.is_dir():
                problems.append(f"missing required bundle directory: {directory.name}")
        manifest_path = layout.manifest
        if not manifest_path.is_file():
            problems.append("missing bundle manifest.json")
        checksums_path = layout.checksums
        if not checksums_path.is_file():
            problems.append("missing bundle checksums.json")
        else:
            checksums = tuple(
                ChecksumRecord.model_validate(entry)
                for entry in json.loads(checksums_path.read_text(encoding="utf-8"))
            )
            for path in sorted(destination.rglob("*")):
                if path.is_file() and path.name != layout.checksums.name:
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
    results_root: Path,
) -> ResultsVerification:
    """Verify one results bundle's structure and checksums."""
    return ResultsVerifier().verify(
        campaign_id,
        results_root=results_root,
    )
