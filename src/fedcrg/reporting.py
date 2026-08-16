"""Builds publication tables/figures, run and repository reports, and immutable results bundles from run artifacts."""

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
from fedcrg.evidence.contracts import ExperimentArtifactContract
from fedcrg.evidence.models import (
    ChecksumRecord,
    EmpiricalExperimentResult,
    EmpiricalPolicyResult,
    ExperimentProvenance,
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
    ExperimentResultsBundleLayout,
    LayoutArtifact,
    LayoutDirectory,
    OutputsLayout,
    PublicationLayout,
    ResultsBundleLayout,
    RunLayout,
    experiment_results_root,
)
from fedcrg.experiments.analyses import (
    BenchmarkReport,
    SyntheticExperimentEnvelope,
    confirmatory_contrasts,
    load_federation_results,
    split_sensitivity,
)
from fedcrg.thresholding.metrics import FederationMetrics
from fedcrg.types import (
    Assurance,
    CalibrationSeed,
    ClientId,
    ConfidenceLevel,
    DataIntegrityError,
    DataRole,
    DecisionSource,
    DecisionState,
    Description,
    DetectorId,
    CompletionState,
    DatasetId,
    ExperimentAxisId,
    ExperimentId,
    ExperimentStatus,
    JsonResultKind,
    ResultsGenerationStatus,
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
    """Filenames of the publication-ready result tables."""

    PROTOCOL_CONSTANTS = "table_1_protocol_constants.csv"
    DATASET_INVENTORY = "table_2_dataset_inventory.csv"
    PRIMARY_POLICY_RESULTS = "table_3_primary_policy_results.csv"
    ADMISSION_STATES = "table_4_admission_states.csv"
    ABLATIONS = "table_5_ablations.csv"
    EXTERNAL_REPLICATION = "table_8_external_replication.csv"


class RepositoryReportFilename(StrEnum):
    """Filenames written into the repository reproducibility report."""

    POLICY_FEDERATION_RESULTS = "policy_federation_results.csv"
    PRIMARY_CONTRASTS = "primary_contrasts.json"
    SPLIT_SENSITIVITY = "split_sensitivity.csv"
    README = "README.md"


_RUN_SUMMARY_FILENAME = LayoutArtifact.RUN_SUMMARY


def _run_manifest(run_dir: Path) -> RunManifest | None:
    manifest_path = RunLayout(run_dir).manifest
    if not manifest_path.is_file():
        return None
    try:
        return RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except pydantic.ValidationError:
        # An unreadable manifest is treated as an incomplete run rather than an error.
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
    """Write a human-readable Markdown summary of one run's manifest, verification status, and federation metrics."""
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
        f"- Experiment: `{manifest.experiment_id}`",
        f"- Policy: `{manifest.policy_id}`",
        f"- Status: `{manifest.status}`",
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
    return Study.load().catalogue.spec(manifest.experiment_id)


class AdmissionStateRow(BaseModel):
    """One client's admission decision and threshold values for the publication admission-states table."""

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
    """One metric's method-vs-comparator contrast for the publication ablations table."""

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
    """One named constant value for the publication protocol-constants table."""

    model_config = Frozen

    constant: Identifier
    value: ProtocolConstantValue


class DatasetInventoryRow(BaseModel):
    """One client's feature count and per-role row counts for the publication dataset-inventory table."""

    model_config = Frozen

    client_id: ClientId
    feature_count: PositiveCount
    role_counts: tuple[RoleCountRow, ...]


class RoleCountRow(BaseModel):
    """Row count and row-id hash for one data role of one client."""

    model_config = Frozen

    role: DataRole
    rows: NonNegativeCount
    row_id_sha256: Sha256


class FederationResultsRow(BaseModel):
    """One run's federation-level metrics for the repository results table."""

    model_config = Frozen

    run_id: RunId
    mebe: Metric
    high_excess: Metric
    band_violation_rate: Fraction
    attack_balanced_macro_tpr: Tpr | None


class PublicationTableBuilder:
    """Builds the individual publication CSV tables from run artifacts."""

    def primary_policy_results(self, run_dirs: tuple[Path, ...], output: Path) -> Path:
        """Write the primary policy results table from completed run directories."""
        records = load_federation_results(run_dirs)
        rows = [record.model_dump(mode="json") for record in records]
        return self._write(pd.DataFrame.from_records(rows), output)

    def admission_states_from_runs(self, run_dirs: tuple[Path, ...], output: Path) -> Path:
        """Write the admission-states table from completed runs' threshold records."""
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
        """Write the ablations table of confirmatory contrasts for the primary experiment's runs."""
        records = load_federation_results(run_dirs)
        primary = tuple(row for row in records if row.experiment_id is ExperimentId.PRIMARY_NBAIOT)
        contrasts = confirmatory_contrasts(
            primary,
            named_calibration_seed=int(config.dataset.primary_calibration_seed),
            expected_model_seeds=tuple(config.randomness.model_seeds),
            bootstrap_seed=config.statistics.bootstrap_seed,
            bootstrap_replicates=config.statistics.bootstrap_replicates,
            bootstrap_ci_percentiles=config.statistics.bootstrap_ci_percentiles,
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
        """Write the protocol-constants table from `config`."""
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
        """Write the dataset-inventory table from a prepared dataset manifest."""
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
        """Write the repository federation-results table from completed run directories."""
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
    """The publication tables and figures produced for a campaign, with a manifest path."""

    tables: tuple[PublicationArtifact, ...]
    figures: tuple[PublicationArtifact, ...]
    manifest: Path

    @property
    def complete(self) -> bool:
        """Whether every table and figure in the package was built successfully."""
        return all(item.available for item in (*self.tables, *self.figures))


@dataclass(frozen=True, slots=True)
class PublicationArtifact:
    """One publication table or figure: whether it was built, and why not if it wasn't."""

    name: Description
    path: Path | None
    available: bool
    reason: Description | None = None


class PublicationPackageBuilder:
    """Builds the full publication package (tables and figures) and its manifest."""

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
        """Build every publication table and figure, recording which succeeded, and write the package manifest."""
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
                    config, table_root / PublicationTableFilename.PROTOCOL_CONSTANTS
                ),
            ),
            self._optional_table(
                "Table 2 - Dataset inventory",
                prepared_manifest,
                lambda path: self.tables.dataset_inventory(
                    path, table_root / PublicationTableFilename.DATASET_INVENTORY
                ),
                "prepared dataset manifest is unavailable",
            ),
            self._runs_table(
                "Table 3 - Primary policy results",
                primary_runs,
                lambda: self.tables.primary_policy_results(
                    primary_runs, table_root / PublicationTableFilename.PRIMARY_POLICY_RESULTS
                ),
                "primary policy runs are unavailable",
            ),
            self._runs_table(
                "Table 4 - Admission states",
                fedcrg_primary,
                lambda: self.tables.admission_states_from_runs(
                    fedcrg_primary, table_root / PublicationTableFilename.ADMISSION_STATES
                ),
                "FedCRG admission runs are unavailable",
            ),
            self._runs_table(
                "Table 5 - Ablations",
                fedcrg_primary,
                lambda: self.tables.ablations(
                    primary_runs, table_root / PublicationTableFilename.ABLATIONS, config
                ),
                "primary policy runs are unavailable",
            ),
        )
        figures = (
            self._table(
                "Figure 1 - Decision architecture",
                lambda: build_decision_architecture_figure(
                    figure_root / "figure_1_decision_architecture.png"
                ),
            ),
            self._table(
                "Figure 2 - Finite-sample readiness frontier",
                lambda: build_readiness_frontier_from_catalogue(
                    figure_root / "figure_2_readiness_frontier.png"
                ),
            ),
            self._table(
                "Figure 3 - Reference-mismatch evidence/power map",
                lambda: build_mismatch_power_map_from_catalogue(
                    figure_root / "figure_3_mismatch_power_map.png"
                ),
            ),
            self._table(
                "Figure 4 - Per-client operating points",
                lambda: build_per_client_operating_points_figure(
                    figure_root / "figure_4_per_client_operating_points.png",
                    _require_bundle_table(
                        destination, PublicationTableFilename.PRIMARY_POLICY_RESULTS
                    ),
                ),
            ),
            self._table(
                "Figure 5 - Reliability-utility frontier",
                lambda: build_reliability_utility_frontier_figure(
                    figure_root / "figure_5_reliability_utility_frontier.png",
                    _require_bundle_table(
                        destination, PublicationTableFilename.PRIMARY_POLICY_RESULTS
                    ),
                ),
            ),
            self._table(
                "Figure 6 - Calibration-size phase transition",
                lambda: build_phase_transition_from_catalogue(
                    figure_root / "figure_6_phase_transition.png"
                ),
            ),
            self._table(
                "Figure 7 - Assumption stress",
                lambda: build_assumption_stress_figure(
                    figure_root / "figure_7_assumption_stress.png",
                    _assumption_stress_frame(destination),
                ),
            ),
            self._table(
                "Figure 8 - External replication",
                lambda: build_external_replication_figure(
                    figure_root / "figure_8_external_replication.png",
                    _require_bundle_table(
                        destination, PublicationTableFilename.EXTERNAL_REPLICATION
                    ),
                ),
            ),
        )
        manifest_path = publication.manifest
        atomic_write_json(
            manifest_path,
            {
                ExperimentStatus.COMPLETE: all(item.available for item in (*tables, *figures)),
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
        """Catches a builder's failure and records it as an unavailable artifact instead of aborting the rest of the package."""
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


def build_decision_architecture_figure(output: Path) -> Path:
    """Render the FedCRG decision-flow diagram."""
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
    """Plot in-band coverage probability against calibration sample count."""
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
    """Plot mismatch-declaration probability against sample count and true reference FPR."""
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
    """Plot the readiness and mismatch-evidence curves side by side."""
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
    path = ResultsBundleLayout(results_root).tables / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Results-driven figure requires the {filename} table, run the campaign first ({path})"
        )
    return pd.read_csv(path)


def _band_guide_lines() -> tuple[Fpr, Fpr, Fpr]:
    protocol = Study.load().study_config.protocol
    return protocol.band.lower, protocol.alpha, protocol.band.upper


def build_per_client_operating_points_figure(output: Path, frame: pd.DataFrame) -> Path:
    """Plot each client's final-test FPR by policy, against the operating band."""
    if frame.empty:
        raise ValueError("operating-points figure requires a non-empty results table")
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


def build_reliability_utility_frontier_figure(output: Path, frame: pd.DataFrame) -> Path:
    """Plot mean excess band error against attack-balanced macro TPR for each policy."""
    if frame.empty:
        raise ValueError("reliability-utility figure requires a non-empty results table")
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


def build_assumption_stress_figure(output: Path, frame: pd.DataFrame) -> Path:
    """Plot realized in-band coverage for each synthetic stress cell."""
    if frame.empty or "coverage" not in frame.columns:
        raise ValueError("assumption-stress figure requires a coverage table")
    combined = frame.copy()
    if "cell" not in combined.columns:
        combined = combined.assign(cell=combined.index.astype(str))
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


def build_external_replication_figure(output: Path, frame: pd.DataFrame) -> Path:
    """Plot each eligible external-replication client's final-test FPR against the operating band."""
    if frame.empty:
        raise ValueError("external-replication figure requires a non-empty results table")
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
    return tuple(Study.load().catalogue.spec(spec_id).axis(axis_id).values)


def build_readiness_frontier_from_catalogue(output: Path) -> Path:
    """Build the readiness-frontier figure using sample counts declared in the experiment catalogue."""
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
    """Build the mismatch-power-map figure using sample counts and true FPRs declared in the experiment catalogue."""
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
    """Build the phase-transition figure using sample counts declared in the experiment catalogue."""
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
    """Build the repository reproducibility report (federation results, confirmatory contrasts, split sensitivity) from run artifacts."""
    reports_root = OutputsLayout(outputs_root).reports / LayoutDirectory.LATEST
    reports_root.mkdir(parents=True, exist_ok=True)
    run_dirs = _completed_runs(outputs_root)
    builder = PublicationTableBuilder()
    policy_table = builder.federation_results(
        run_dirs, reports_root / RepositoryReportFilename.POLICY_FEDERATION_RESULTS
    )
    records = load_federation_results(run_dirs)
    primary_records = tuple(
        row for row in records if row.experiment_id is ExperimentId.PRIMARY_NBAIOT
    )
    contrasts_path = reports_root / RepositoryReportFilename.PRIMARY_CONTRASTS
    if primary_records:
        contrasts = confirmatory_contrasts(
            primary_records,
            named_calibration_seed=int(config.dataset.primary_calibration_seed),
            expected_model_seeds=tuple(config.randomness.model_seeds),
            bootstrap_seed=config.statistics.bootstrap_seed,
            bootstrap_replicates=config.statistics.bootstrap_replicates,
            bootstrap_ci_percentiles=config.statistics.bootstrap_ci_percentiles,
        )
        contrasts_payload = [
            {
                "comparator": item.comparator,
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
    sensitivity_path = reports_root / RepositoryReportFilename.SPLIT_SENSITIVITY
    pd.DataFrame.from_records(
        [
            row.model_dump(mode="json")
            for row in split_sensitivity(
                primary_records, config.statistics.split_sensitivity_percentiles
            )
        ]
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
    output = reports_root / RepositoryReportFilename.README
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def build_publication(
    config: ExperimentConfig,
    outputs_root: Path,
    prepared_manifest: Path | None = None,
    destination: Path | None = None,
) -> Path:
    """Build the full publication package and return its manifest path."""
    package = PublicationPackageBuilder().build(
        config=config,
        outputs_root=outputs_root,
        prepared_manifest=prepared_manifest,
        destination=destination,
    )
    return package.manifest


def _assumption_stress_frame(results_root: Path) -> pd.DataFrame:
    statistics_root = ResultsBundleLayout(results_root).statistics
    if not statistics_root.is_dir():
        raise FileNotFoundError(
            f"Assumption-stress figure requires campaign statistics ({statistics_root})"
        )
    frames: list[pd.DataFrame] = []
    for path in sorted(statistics_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "coverage" in payload:
            frames.append(pd.DataFrame([{"cell": path.stem, "coverage": payload["coverage"]}]))
    if not frames:
        raise FileNotFoundError(f"No coverage statistics found under {statistics_root}")
    return pd.concat(frames, ignore_index=True)


class OperatingPointRow(BaseModel):
    """One client/policy false-positive rate used to render an operating-points figure."""

    model_config = Frozen

    client_id: ClientId
    fpr: Fpr
    policy_id: PolicyId


class ExperimentPublication(BaseModel):
    """Paths written by publishing one experiment's delivery artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    experiment_id: ExperimentId
    json_path: Path
    csv_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    report_path: Path


class ExperimentPublisher:
    """Writes experiment-owned JSON, tables, figures, and reports from verified execution artifacts."""

    def publish(
        self, experiment_id: ExperimentId, outputs_root: Path, config: ExperimentConfig
    ) -> ExperimentPublication:
        """Materialize the experiment's required publication artifacts from existing execution evidence."""
        from fedcrg.evidence.contracts import experiment_contract

        contract = experiment_contract(experiment_id)
        outputs = OutputsLayout(outputs_root)
        json_path = self._write_json(contract.json_kind, experiment_id, outputs, config)
        csv_paths = self._write_csv(contract, experiment_id, outputs, json_path)
        figure_paths = self._write_figures(contract, experiment_id, outputs, json_path, csv_paths)
        report_path = self._write_report(contract, experiment_id, outputs, json_path)
        return ExperimentPublication(
            experiment_id=experiment_id,
            json_path=json_path,
            csv_paths=csv_paths,
            figure_paths=figure_paths,
            report_path=report_path,
        )

    def _write_json(
        self,
        kind: JsonResultKind,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
        config: ExperimentConfig,
    ) -> Path:
        destination = (
            outputs.experiment_reports(experiment_id) / LayoutArtifact.EXPERIMENT_RESULT_JSON
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if kind is JsonResultKind.EMPIRICAL_RESULTS:
            payload = self._empirical_result(experiment_id, outputs, config)
            atomic_write_json(destination, payload)
            return destination
        source = outputs.analysis_result(experiment_id)
        if not source.is_file():
            raise FileNotFoundError(f"Missing analysis JSON for {experiment_id}: {source}")
        if kind is JsonResultKind.SYNTHETIC_ENVELOPE:
            envelope = SyntheticExperimentEnvelope.model_validate_json(
                source.read_text(encoding="utf-8")
            )
            if not envelope.cells:
                raise ValueError(f"{experiment_id} analysis JSON contains no cells")
            atomic_write_json(destination, envelope)
            return destination
        report = BenchmarkReport.model_validate_json(source.read_text(encoding="utf-8"))
        if not report.cells:
            raise ValueError(f"{experiment_id} benchmark JSON contains no cells")
        atomic_write_json(destination, report)
        return destination

    @staticmethod
    def _empirical_result(
        experiment_id: ExperimentId, outputs: OutputsLayout, config: ExperimentConfig
    ) -> EmpiricalExperimentResult:
        if not outputs.runs.is_dir():
            raise FileNotFoundError(f"No run directories exist for {experiment_id}")
        records: list[EmpiricalPolicyResult] = []
        for run_dir in sorted(path for path in outputs.runs.iterdir() if path.is_dir()):
            layout = RunLayout(run_dir)
            if not layout.manifest.is_file() or not layout.federation_metrics.is_file():
                continue
            try:
                manifest = RunManifest.model_validate_json(
                    layout.manifest.read_text(encoding="utf-8")
                )
            except pydantic.ValidationError:
                continue
            if (
                manifest.experiment_id is not experiment_id
                or manifest.status is not ExperimentStatus.COMPLETE
                or manifest.config_hash != config.config_hash
            ):
                continue
            federation = FederationMetrics.model_validate_json(
                layout.federation_metrics.read_text(encoding="utf-8")
            )
            records.append(
                EmpiricalPolicyResult(
                    run_id=manifest.run_id,
                    policy_id=manifest.policy_id,
                    model_seed=manifest.model_seed,
                    calibration_seed=manifest.calibration_seed,
                    config_hash=manifest.config_hash,
                    mebe=federation.mebe,
                    high_excess=federation.high_excess,
                    band_violation_rate=federation.band_violation_rate,
                    attack_balanced_macro_tpr=federation.attack_balanced_macro_tpr,
                )
            )
        if not records:
            raise ValueError(f"{experiment_id} has no completed federation metrics to publish")
        return EmpiricalExperimentResult(
            experiment_id=experiment_id,
            config_hash=config.config_hash,
            records=tuple(records),
        )

    def _write_csv(
        self,
        contract: ExperimentArtifactContract,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
        json_path: Path,
    ) -> tuple[Path, ...]:
        if not contract.csv_files:
            return ()
        table_root = outputs.experiment_tables(experiment_id)
        table_root.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for item in contract.csv_files:
            destination = table_root / item.filename
            if "cells" in payload:
                frame = pd.DataFrame.from_records(payload["cells"])
            elif "records" in payload:
                frame = pd.DataFrame.from_records(payload["records"])
            else:
                raise ValueError(f"{experiment_id} JSON result is not tabular")
            if frame.empty:
                raise ValueError(f"{experiment_id} produced an empty CSV")
            frame.to_csv(destination, index=False)
            written.append(destination)
        return tuple(written)

    def _write_figures(
        self,
        contract: ExperimentArtifactContract,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
        json_path: Path,
        csv_paths: tuple[Path, ...],
    ) -> tuple[Path, ...]:
        if not contract.figure_files:
            return ()
        figure_root = outputs.experiment_figures(experiment_id)
        figure_root.mkdir(parents=True, exist_ok=True)
        frame = pd.read_csv(csv_paths[0]) if csv_paths else pd.DataFrame()
        written: list[Path] = []
        for item in contract.figure_files:
            destination = figure_root / item.filename
            if item.filename is LayoutArtifact.EXPERIMENT_POWER_FIGURE:
                self._coverage_or_power_figure(destination, frame, power=True)
            elif item.filename is LayoutArtifact.EXPERIMENT_COVERAGE_FIGURE:
                self._coverage_or_power_figure(destination, frame, power=False)
            elif item.filename is LayoutArtifact.EXPERIMENT_OPERATING_POINTS_FIGURE:
                self._operating_points_from_runs(destination, experiment_id, outputs)
            elif item.filename is LayoutArtifact.EXPERIMENT_RELIABILITY_FIGURE:
                build_reliability_utility_frontier_figure(destination, frame)
            elif item.filename is LayoutArtifact.EXPERIMENT_REPLICATION_FIGURE:
                self._replication_figure(destination, experiment_id, outputs, frame)
            else:
                raise ValueError(f"Unsupported figure {item.filename} for {experiment_id}")
            written.append(destination)
        return tuple(written)

    @staticmethod
    def _coverage_or_power_figure(output: Path, frame: pd.DataFrame, *, power: bool) -> Path:
        if frame.empty:
            raise ValueError("Figure requires a non-empty results table")
        figure, axis = plt.subplots(figsize=(8, 5))
        if power and "declaration_probability" in frame.columns:
            if {"sample_count", "true_fpr", "declaration_probability"}.issubset(frame.columns):
                pivot = frame.pivot_table(
                    index="sample_count",
                    columns="true_fpr",
                    values="declaration_probability",
                    aggfunc="mean",
                )
                image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower")
                figure.colorbar(image, ax=axis, label="declaration probability")
                axis.set_xlabel("true FPR")
                axis.set_ylabel("sample count")
            else:
                axis.plot(frame["declaration_probability"], marker="o")
                axis.set_ylabel("declaration probability")
        else:
            y_column = next(
                (
                    name
                    for name in ("empirical_probability", "coverage", "declaration_probability")
                    if name in frame.columns
                ),
                None,
            )
            if y_column is None:
                raise ValueError("Coverage figure requires a probability column")
            x_column = next(
                (name for name in ("sample_count", "value", "true_fpr") if name in frame.columns),
                None,
            )
            if x_column is None:
                axis.plot(frame[y_column], marker="o")
            else:
                axis.plot(frame[x_column], frame[y_column], marker="o")
                axis.set_xlabel(x_column)
            axis.set_ylabel(y_column)
        axis.set_title("Experiment result")
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(figure)
        if output.stat().st_size == 0:
            raise RuntimeError(f"Figure writer produced an empty file: {output}")
        return output

    @staticmethod
    def _operating_points_from_runs(
        output: Path, experiment_id: ExperimentId, outputs: OutputsLayout
    ) -> Path:
        rows: list[OperatingPointRow] = []
        if outputs.runs.is_dir():
            for run_dir in outputs.runs.iterdir():
                if not run_dir.is_dir():
                    continue
                layout = RunLayout(run_dir)
                if not layout.manifest.is_file() or not layout.metric_records.is_file():
                    continue
                try:
                    manifest = RunManifest.model_validate_json(
                        layout.manifest.read_text(encoding="utf-8")
                    )
                except pydantic.ValidationError:
                    continue
                if manifest.experiment_id is not experiment_id:
                    continue
                for line in layout.metric_records.read_text(encoding="utf-8").splitlines():
                    if not line:
                        continue
                    try:
                        record = MetricRecord.model_validate_json(line)
                    except pydantic.ValidationError:
                        continue
                    rows.append(
                        OperatingPointRow(
                            client_id=record.client_id,
                            fpr=record.fpr,
                            policy_id=record.policy_id,
                        )
                    )
        if not rows:
            raise ValueError("Operating-points figure requires per-client metric records")
        return build_per_client_operating_points_figure(
            output, pd.DataFrame.from_records([row.model_dump(mode="json") for row in rows])
        )

    @staticmethod
    def _replication_figure(
        output: Path,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
        frame: pd.DataFrame,
    ) -> Path:
        if {"client_id", "fpr"}.issubset(frame.columns):
            return build_external_replication_figure(output, frame)
        return ExperimentPublisher._operating_points_from_runs(output, experiment_id, outputs)

    @staticmethod
    def _write_report(
        contract: ExperimentArtifactContract,
        experiment_id: ExperimentId,
        outputs: OutputsLayout,
        json_path: Path,
    ) -> Path:
        destination = outputs.experiment_reports(experiment_id) / LayoutArtifact.EXPERIMENT_REPORT
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        count = len(payload.get("cells") or payload.get("records") or ())
        lines = [
            f"# FedCRG experiment {experiment_id}",
            "",
            f"- JSON result: `{json_path.name}`",
            f"- Result rows: `{count}`",
            f"- Category: `{contract.category}`",
            "",
            "This report is generated from the experiment's machine-readable result payload.",
        ]
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination


class ResultsManifest(BaseModel):
    """Top-level manifest describing a results bundle's provenance and completeness."""

    model_config = Frozen

    experiment_id: ExperimentId | None = None
    complete: bool
    completion_state: CompletionState
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
    """Result of checking a results bundle's structure and checksums."""

    valid: bool
    problems: tuple[Description, ...]


@dataclass(frozen=True, slots=True)
class ResultsBuildResult:
    """Outcome of building or reusing a results bundle."""

    path: Path
    status: ResultsGenerationStatus


class ResultsBuilder:
    """Assembles a delivery results bundle from verified execution artifacts."""

    def build(
        self,
        *,
        outputs_root: Path,
        results_root: Path,
        experiment_id: ExperimentId | None = None,
        overwrite: bool = False,
        study: Study | None = None,
    ) -> Path:
        """Assemble the campaign bundle, or one experiment bundle under results/experiments/."""
        return self.build_with_status(
            outputs_root=outputs_root,
            results_root=results_root,
            experiment_id=experiment_id,
            overwrite=overwrite,
            study=study,
        ).path

    def build_with_status(
        self,
        *,
        outputs_root: Path,
        results_root: Path,
        experiment_id: ExperimentId | None = None,
        overwrite: bool = False,
        study: Study | None = None,
    ) -> ResultsBuildResult:
        """Assemble a bundle, reusing a current valid one unless `overwrite` is set."""
        resolved = study or Study.load()
        if experiment_id is None:
            return self._build_campaign(
                outputs_root, results_root, overwrite=overwrite, study=resolved
            )
        return self._build_experiment(
            experiment_id, outputs_root, results_root, overwrite=overwrite, study=resolved
        )

    def _build_campaign(
        self, outputs_root: Path, results_root: Path, *, overwrite: bool, study: Study
    ) -> ResultsBuildResult:
        destination = results_root
        layout = ResultsBundleLayout(destination)
        if layout.manifest.is_file() and not overwrite:
            verification = ResultsVerifier().verify(results_root)
            if verification.valid:
                return ResultsBuildResult(destination, ResultsGenerationStatus.ALREADY_GENERATED)
        if overwrite:
            self._clear_campaign_delivery(layout)
        existed = layout.manifest.is_file()
        for directory in layout.required_directories:
            directory.mkdir(parents=True, exist_ok=True)

        config = study.resolve(ExperimentId.PRIMARY_NBAIOT)
        self._write_resolved_configs(config, layout)
        self._copy_metrics(outputs_root, layout)
        self._copy_statistics(outputs_root, layout)
        self._write_reports(outputs_root, config, layout)
        self._write_provenance(outputs_root, config, layout)
        self._copy_tables_and_figures(outputs_root, layout)

        complete = self._evidence_complete(study, outputs_root)
        self._write_manifest_and_checksums(
            layout,
            config,
            outputs_root,
            experiment_id=None,
            complete=complete,
            completion_state=CompletionState.FULLY_PASSED
            if complete
            else CompletionState.EVIDENCE_INCOMPLETE,
        )
        return ResultsBuildResult(
            destination,
            ResultsGenerationStatus.REBUILT
            if existed or overwrite
            else ResultsGenerationStatus.BUILT,
        )

    def _build_experiment(
        self,
        experiment_id: ExperimentId,
        outputs_root: Path,
        results_root: Path,
        *,
        overwrite: bool,
        study: Study,
    ) -> ResultsBuildResult:
        destination = experiment_results_root(results_root, experiment_id)
        layout = ExperimentResultsBundleLayout(destination)
        if layout.manifest.is_file() and not overwrite:
            verification = ResultsVerifier().verify(results_root, experiment_id=experiment_id)
            if verification.valid:
                return ResultsBuildResult(destination, ResultsGenerationStatus.ALREADY_GENERATED)
        existed = layout.manifest.is_file()
        if destination.exists() and (overwrite or existed):
            import shutil

            shutil.rmtree(destination)
        for directory in layout.required_directories:
            directory.mkdir(parents=True, exist_ok=True)

        config = study.resolve(experiment_id)
        layout.resolved_configs.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            layout.resolved_configs / f"{experiment_id}.json",
            {
                "config_hash": config.config_hash,
                "data_spec_hash": config.data_spec_hash,
                "payload": config.model_dump(mode="json"),
            },
        )
        self._copy_experiment_delivery(experiment_id, outputs_root, layout)
        self._copy_metrics(outputs_root, layout, experiment_id=experiment_id)
        self._copy_run_artifacts(outputs_root, layout, experiment_id=experiment_id)
        self._write_experiment_provenance(experiment_id, outputs_root, config, layout)
        from fedcrg.evidence.completion import ExperimentEvidenceAssessor
        from fedcrg.evidence.contracts import experiment_contract
        from fedcrg.evidence.completion import experiment_source_files

        contract = experiment_contract(experiment_id)
        published = experiment_source_files(contract, outputs_root, experiment_id)
        source_ok = bool(published) and all(
            path.is_file() and path.stat().st_size > 0 for path in published
        )
        assessment = ExperimentEvidenceAssessor(study).assess(
            experiment_id, outputs_root=outputs_root, results_root=results_root
        )
        blocking = {
            CompletionState.NOT_STARTED,
            CompletionState.RUNNING,
            CompletionState.FAILED,
            CompletionState.INVALID,
            CompletionState.EXECUTION_INCOMPLETE,
        }
        complete = source_ok and assessment.state not in blocking
        self._write_manifest_and_checksums(
            layout,
            config,
            outputs_root,
            experiment_id=experiment_id,
            complete=complete,
            completion_state=CompletionState.ANALYSIS_COMPLETE if complete else assessment.state,
        )
        return ResultsBuildResult(
            destination,
            ResultsGenerationStatus.REBUILT
            if existed or overwrite
            else ResultsGenerationStatus.BUILT,
        )

    @staticmethod
    def _write_manifest_and_checksums(
        layout: ResultsBundleLayout | ExperimentResultsBundleLayout,
        config: ExperimentConfig,
        outputs_root: Path,
        *,
        experiment_id: ExperimentId | None,
        complete: bool,
        completion_state: CompletionState,
    ) -> None:
        """Write the bundle manifest first, then checksums that also cover it."""
        exclude_experiments = experiment_id is None
        checksums = ResultsBuilder._checksums(layout, exclude_experiments=exclude_experiments)
        manifest = ResultsBuilder._manifest(
            config,
            outputs_root,
            checksums,
            complete,
            experiment_id=experiment_id,
            file_count=len(checksums) + 1,
            completion_state=completion_state,
        )
        atomic_write_json(layout.manifest, manifest)
        checksums = ResultsBuilder._checksums(layout, exclude_experiments=exclude_experiments)
        atomic_write_json(layout.checksums, checksums)

    @staticmethod
    def _clear_campaign_delivery(layout: ResultsBundleLayout) -> None:
        for path in (layout.manifest, layout.checksums):
            if path.is_file():
                path.unlink()
        for directory in layout.required_directories:
            if directory.is_dir():
                import shutil

                shutil.rmtree(directory)

    @staticmethod
    def _copy_experiment_delivery(
        experiment_id: ExperimentId,
        outputs_root: Path,
        layout: ExperimentResultsBundleLayout,
    ) -> None:
        from fedcrg.evidence.contracts import experiment_contract

        contract = experiment_contract(experiment_id)
        outputs = OutputsLayout(outputs_root)
        copies = (
            (
                outputs.experiment_reports(experiment_id) / LayoutArtifact.EXPERIMENT_RESULT_JSON,
                layout.json_dir / LayoutArtifact.EXPERIMENT_RESULT_JSON,
            ),
            *(
                (
                    outputs.experiment_tables(experiment_id) / item.filename,
                    layout.csv_dir / item.filename,
                )
                for item in contract.csv_files
            ),
            *(
                (
                    outputs.experiment_figures(experiment_id) / item.filename,
                    layout.figures / item.filename,
                )
                for item in contract.figure_files
            ),
            *(
                (
                    outputs.experiment_reports(experiment_id) / item.filename,
                    layout.reports / item.filename,
                )
                for item in contract.report_files
            ),
        )
        for source, destination in copies:
            if not source.is_file():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

    @staticmethod
    def _write_experiment_provenance(
        experiment_id: ExperimentId,
        outputs_root: Path,
        config: ExperimentConfig,
        layout: ExperimentResultsBundleLayout,
    ) -> None:
        from fedcrg.evidence.completion import source_digests
        from fedcrg.evidence.contracts import experiment_contract

        contract = experiment_contract(experiment_id)
        provenance = ExperimentProvenance(
            experiment_id=experiment_id,
            config_hash=config.config_hash,
            data_spec_hash=config.data_spec_hash,
            source_digests=source_digests(contract, outputs_root, experiment_id),
        )
        atomic_write_json(layout.provenance_json, provenance)

    @staticmethod
    def _evidence_complete(study: Study, outputs_root: Path) -> bool:
        from fedcrg.evidence.completion import ExperimentEvidenceAssessor
        from fedcrg.evidence.contracts import experiment_contract

        assessor = ExperimentEvidenceAssessor(study)
        for spec in study.catalogue.all():
            contract = experiment_contract(spec.id)
            if contract.optional or contract.inapplicable:
                continue
            if not assessor.assess(
                spec.id,
                outputs_root=outputs_root,
                results_root=study.paths.results_root,
            ).passed:
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
    def _copy_metrics(
        outputs_root: Path,
        layout: ResultsBundleLayout | ExperimentResultsBundleLayout,
        *,
        experiment_id: ExperimentId | None = None,
    ) -> None:
        runs_root = OutputsLayout(outputs_root).runs
        if not runs_root.exists():
            return
        records: list[MetricRecord] = []
        for run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            manifest = _run_manifest(run_root)
            if manifest is None or manifest.status is not ExperimentStatus.COMPLETE:
                continue
            if experiment_id is not None and manifest.experiment_id is not experiment_id:
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
    def _copy_run_artifacts(
        outputs_root: Path, layout: ExperimentResultsBundleLayout, experiment_id: ExperimentId
    ) -> None:
        """Copy each completed run's manifest, verification hashes, and summary report into the bundle."""
        runs_root = OutputsLayout(outputs_root).runs
        if not runs_root.exists():
            return
        for run_root in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            manifest = _run_manifest(run_root)
            if manifest is None or manifest.status is not ExperimentStatus.COMPLETE:
                continue
            if manifest.experiment_id is not experiment_id:
                continue
            source = RunLayout(run_root)
            target = layout.runs / run_root.name
            for artifact in (
                source.manifest,
                source.verification / LayoutArtifact.HASHES,
                source.reports / _RUN_SUMMARY_FILENAME,
            ):
                if not artifact.is_file():
                    continue
                destination = target / artifact.relative_to(run_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(artifact.read_bytes())

    @staticmethod
    def _copy_statistics(outputs_root: Path, layout: ResultsBundleLayout) -> None:
        outputs_layout = OutputsLayout(outputs_root)
        for source, target in (
            (outputs_layout.readiness_plans_file, layout.readiness_plans),
            (outputs_layout.mismatch_cutoffs_file, layout.mismatch_cutoffs),
        ):
            if source.is_file():
                resolved_target = target.resolve()
                # Guards against a symlink or misconfigured layout writing outside the bundle.
                if resolved_target.parent != layout.statistics.resolve():
                    raise DataIntegrityError(
                        f"Statistics artifact escapes the statistics directory: {target}"
                    )
                resolved_target.write_bytes(source.read_bytes())

    def _write_reports(
        self, outputs_root: Path, config: ExperimentConfig, layout: ResultsBundleLayout
    ) -> None:
        report = build_repository_report(outputs_root, config)
        if not report.is_file():
            return
        destination = (layout.reports / report.name).resolve()
        if destination.parent != layout.reports.resolve():
            raise DataIntegrityError(f"Repository report escapes the reports directory: {report}")
        destination.write_bytes(report.read_bytes())

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
    def _checksums(
        layout: ResultsBundleLayout | ExperimentResultsBundleLayout,
        *,
        exclude_experiments: bool,
    ) -> tuple[ChecksumRecord, ...]:
        root = layout.root
        records: list[ChecksumRecord] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == layout.checksums.name:
                continue
            relative = path.relative_to(root).as_posix()
            if exclude_experiments and relative.startswith(f"{LayoutDirectory.EXPERIMENTS}/"):
                continue
            records.append(ChecksumRecord(relative_path=relative, sha256=sha256_file(path)))
        return tuple(records)

    @staticmethod
    def _manifest(
        config: ExperimentConfig,
        outputs_root: Path,
        checksums: tuple[ChecksumRecord, ...],
        complete: bool,
        *,
        experiment_id: ExperimentId | None,
        file_count: PositiveCount | None = None,
        completion_state: CompletionState,
    ) -> ResultsManifest:
        detector = config.detector
        return ResultsManifest(
            experiment_id=experiment_id,
            complete=complete,
            completion_state=completion_state,
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
    """Verifies a previously built results bundle's structure and checksums."""

    def verify(
        self,
        results_root: Path,
        experiment_id: ExperimentId | None = None,
    ) -> ResultsVerification:
        """Check the campaign bundle and every experiment bundle, or one experiment bundle."""
        problems: list[Identifier] = []
        if experiment_id is None:
            if not results_root.exists():
                return ResultsVerification(
                    False, (f"results bundle does not exist: {results_root}",)
                )
            problems.extend(
                self._verify_bundle(ResultsBundleLayout(results_root), exclude_experiments=True)
            )
            experiments_root = results_root / LayoutDirectory.EXPERIMENTS
            if experiments_root.is_dir():
                for bundle_root in sorted(
                    path for path in experiments_root.iterdir() if path.is_dir()
                ):
                    problems.extend(
                        self._verify_bundle(
                            ExperimentResultsBundleLayout(bundle_root),
                            exclude_experiments=False,
                        )
                    )
        else:
            destination = experiment_results_root(results_root, experiment_id)
            if not destination.exists():
                return ResultsVerification(
                    False, (f"experiment results bundle does not exist: {destination}",)
                )
            problems.extend(
                self._verify_bundle(
                    ExperimentResultsBundleLayout(destination), exclude_experiments=False
                )
            )
        return ResultsVerification(not problems, tuple(problems))

    @staticmethod
    def _verify_bundle(
        layout: ResultsBundleLayout | ExperimentResultsBundleLayout,
        *,
        exclude_experiments: bool,
    ) -> list[Identifier]:
        """Check one bundle's required directories, manifest, checksums, and files."""
        destination = layout.root
        problems: list[Identifier] = []
        for directory in layout.required_directories:
            if not directory.is_dir():
                problems.append(f"missing required bundle directory: {directory.name}")
        if not layout.manifest.is_file():
            problems.append("missing bundle manifest.json")
            if not layout.checksums.is_file():
                problems.append("missing bundle checksums.json")
            return problems
        try:
            manifest = ResultsManifest.model_validate_json(
                layout.manifest.read_text(encoding="utf-8")
            )
        except pydantic.ValidationError:
            problems.append("bundle manifest.json failed schema validation")
            return problems
        if layout.manifest.stat().st_size == 0:
            problems.append("bundle manifest.json is empty")
        if not layout.checksums.is_file():
            problems.append("missing bundle checksums.json")
        else:
            problems.extend(
                ResultsVerifier._checksum_mismatches(
                    destination, layout.checksums, exclude_experiments=exclude_experiments
                )
            )
        if isinstance(layout, ExperimentResultsBundleLayout):
            problems.extend(ResultsVerifier._verify_experiment_contents(layout, manifest))
        elif manifest.complete:
            if manifest.completion_state is not CompletionState.FULLY_PASSED:
                problems.append("campaign bundle claims complete without fully_passed state")
        return problems

    @staticmethod
    def _verify_experiment_contents(
        layout: ExperimentResultsBundleLayout, manifest: ResultsManifest
    ) -> list[Identifier]:
        from fedcrg.evidence.contracts import experiment_contract

        if manifest.experiment_id is None:
            return ["experiment bundle manifest is missing experiment_id"]
        contract = experiment_contract(manifest.experiment_id)
        problems: list[Identifier] = []
        if manifest.complete and manifest.completion_state not in {
            CompletionState.FULLY_PASSED,
            CompletionState.ANALYSIS_COMPLETE,
        }:
            problems.append("bundle claims complete without a passed or analysis-complete state")
        required = (
            *((layout.json_dir / item.filename, "json") for item in contract.json_files),
            *((layout.csv_dir / item.filename, "csv") for item in contract.csv_files),
            *((layout.figures / item.filename, "figure") for item in contract.figure_files),
            *((layout.reports / item.filename, "report") for item in contract.report_files),
        )
        for path, kind in required:
            if not path.is_file():
                problems.append(f"missing required {kind} artifact: {path.name}")
            elif path.stat().st_size == 0:
                problems.append(f"empty required {kind} artifact: {path.name}")
            elif kind == "json":
                problems.extend(ResultsVerifier._validate_result_json(path, contract.json_kind))
            elif kind == "csv":
                try:
                    frame = pd.read_csv(path)
                except (OSError, pd.errors.ParserError, UnicodeError):
                    problems.append(f"unreadable CSV artifact: {path.name}")
                else:
                    if frame.empty:
                        problems.append(f"CSV artifact has no data rows: {path.name}")
        if not layout.provenance_json.is_file():
            problems.append("missing experiment bundle provenance")
        else:
            try:
                ExperimentProvenance.model_validate_json(
                    layout.provenance_json.read_text(encoding="utf-8")
                )
            except pydantic.ValidationError:
                problems.append("experiment bundle provenance failed schema validation")
        return problems

    @staticmethod
    def _validate_result_json(path: Path, kind: JsonResultKind) -> list[Identifier]:
        try:
            payload = path.read_text(encoding="utf-8")
            parsed = json.loads(payload)
        except (OSError, json.JSONDecodeError):
            return [f"malformed JSON artifact: {path.name}"]
        if not isinstance(parsed, dict) or not parsed:
            return [f"JSON artifact has no content: {path.name}"]
        try:
            if kind is JsonResultKind.SYNTHETIC_ENVELOPE:
                envelope = SyntheticExperimentEnvelope.model_validate_json(payload)
                if not envelope.cells:
                    return [f"JSON artifact contains no cells: {path.name}"]
            elif kind is JsonResultKind.BENCHMARK_REPORT:
                report = BenchmarkReport.model_validate_json(payload)
                if not report.cells:
                    return [f"JSON artifact contains no cells: {path.name}"]
            else:
                result = EmpiricalExperimentResult.model_validate_json(payload)
                if not result.records:
                    return [f"JSON artifact contains no records: {path.name}"]
        except pydantic.ValidationError:
            return [f"JSON artifact failed typed schema: {path.name}"]
        return []

    @staticmethod
    def _checksum_mismatches(
        destination: Path, checksums_path: Path, *, exclude_experiments: bool
    ) -> list[Identifier]:
        problems: list[Identifier] = []
        try:
            checksums = tuple(
                ChecksumRecord.model_validate(entry)
                for entry in json.loads(checksums_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, pydantic.ValidationError):
            return ["bundle checksums.json failed schema validation"]
        listed = {record.relative_path: record.sha256 for record in checksums}
        seen: set[PathString] = set()
        for path in sorted(destination.rglob("*")):
            if not path.is_file() or path.name == checksums_path.name:
                continue
            relative = path.relative_to(destination).as_posix()
            if exclude_experiments and relative.startswith(f"{LayoutDirectory.EXPERIMENTS}/"):
                continue
            seen.add(relative)
            expected = listed.get(relative)
            if expected is None:
                problems.append(f"unchecksummed bundle file: {relative}")
            elif str(expected) != sha256_file(path):
                problems.append(f"bundle hash mismatch: {relative}")
        for relative in listed:
            if exclude_experiments and relative.startswith(f"{LayoutDirectory.EXPERIMENTS}/"):
                continue
            if relative not in seen:
                problems.append(f"missing checksummed bundle file: {relative}")
        return problems


def build_results_bundle(
    outputs_root: Path, results_root: Path, *, overwrite: bool = False
) -> Path:
    """Build the campaign results bundle at `results_root`."""
    return ResultsBuilder().build(
        outputs_root=outputs_root, results_root=results_root, overwrite=overwrite
    )


def build_experiment_results_bundle(
    experiment_id: ExperimentId,
    outputs_root: Path,
    results_root: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Build the results bundle for one experiment under results/experiments/."""
    return ResultsBuilder().build(
        outputs_root=outputs_root,
        results_root=results_root,
        experiment_id=experiment_id,
        overwrite=overwrite,
    )


def ensure_experiment_results_bundle(
    experiment_id: ExperimentId,
    outputs_root: Path,
    results_root: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Return the experiment's bundle, rebuilding it when missing, invalid, or overwritten."""
    return (
        ResultsBuilder()
        .build_with_status(
            outputs_root=outputs_root,
            results_root=results_root,
            experiment_id=experiment_id,
            overwrite=overwrite,
        )
        .path
    )


def verify_results_bundle(
    results_root: Path,
    experiment_id: ExperimentId | None = None,
) -> ResultsVerification:
    """Verify the campaign bundle and every experiment bundle, or one experiment bundle."""
    return ResultsVerifier().verify(results_root, experiment_id=experiment_id)
