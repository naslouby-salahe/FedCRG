"""Fixed-federation primary contrasts and split-sensitivity analysis from run evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fedcrg.analysis.statistics import (
    DescriptiveSummary,
    PairedBootstrapInterval,
    describe,
    paired_model_seed_bootstrap,
    split_sensitivity_summary,
)
from fedcrg.core.enums import PolicyId


@dataclass(frozen=True, slots=True)
class FederationResultRecord:
    run_id: str
    experiment_id: str
    dataset_id: str
    model_seed: int
    calibration_seed: int
    policy: PolicyId
    mebe: float
    high_excess: float
    band_violation_rate: float
    attack_balanced_macro_tpr: float | None


@dataclass(frozen=True, slots=True)
class ContrastMetricResult:
    metric: str
    method_summary: DescriptiveSummary
    comparator_summary: DescriptiveSummary
    paired_difference: PairedBootstrapInterval
    relative_difference: float | None


@dataclass(frozen=True, slots=True)
class PolicyContrastResult:
    comparator: PolicyId
    metrics: tuple[ContrastMetricResult, ...]


def load_federation_results(run_dirs: tuple[Path, ...]) -> tuple[FederationResultRecord, ...]:
    rows: list[FederationResultRecord] = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        federation_path = run_dir / "metrics" / "federation.json"
        config_path = run_dir / "run_config.json"
        if (
            not manifest_path.is_file()
            or not federation_path.is_file()
            or not config_path.is_file()
        ):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        federation = json.loads(federation_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            continue
        rows.append(
            FederationResultRecord(
                run_id=str(manifest["run_id"]),
                experiment_id=str(manifest["experiment_id"]),
                dataset_id=str(config["parameters"]["dataset"]["id"]),
                model_seed=int(manifest["model_seed"]),
                calibration_seed=int(manifest["calibration_seed"]),
                policy=PolicyId(str(manifest["policy_id"])),
                mebe=float(federation["mebe"]),
                high_excess=float(federation["high_excess"]),
                band_violation_rate=float(federation["band_violation_rate"]),
                attack_balanced_macro_tpr=(
                    None
                    if federation.get("attack_balanced_macro_tpr") is None
                    else float(federation["attack_balanced_macro_tpr"])
                ),
            )
        )
    return tuple(rows)


def confirmatory_contrasts(
    records: tuple[FederationResultRecord, ...],
    *,
    named_calibration_seed: int,
    bootstrap_seed: int = 424242,
    bootstrap_replicates: int = 10_000,
) -> tuple[PolicyContrastResult, ...]:
    """Compute exactly the four pre-registered method-minus-comparator contrasts."""

    comparators = (
        PolicyId.GLOBAL_QUANTILE,
        PolicyId.LOCAL_QUANTILE,
        PolicyId.READINESS_ONLY,
        PolicyId.SHRINKAGE,
    )
    selected = tuple(row for row in records if row.calibration_seed == named_calibration_seed)
    expected_seeds = {11, 22, 33, 44, 55}
    observed_method_seeds = {row.model_seed for row in selected if row.policy is PolicyId.FEDCRG}
    if observed_method_seeds != expected_seeds:
        raise ValueError(
            "Confirmatory contrasts require exactly model seeds 11,22,33,44,55 "
            f"for FedCRG, observed {sorted(observed_method_seeds)}"
        )
    method_by_seed = {row.model_seed: row for row in selected if row.policy is PolicyId.FEDCRG}
    results: list[PolicyContrastResult] = []
    for comparator in comparators:
        comparator_by_seed = {row.model_seed: row for row in selected if row.policy is comparator}
        if set(comparator_by_seed) != expected_seeds:
            raise ValueError(
                f"Confirmatory comparator {comparator.value} is missing paired model seeds"
            )
        common_seeds = tuple(sorted(expected_seeds))
        metrics: list[ContrastMetricResult] = []
        for metric in ("mebe", "high_excess", "attack_balanced_macro_tpr"):
            left_values = tuple(getattr(method_by_seed[seed], metric) for seed in common_seeds)
            right_values = tuple(getattr(comparator_by_seed[seed], metric) for seed in common_seeds)
            if any(value is None for value in left_values + right_values):
                continue
            left = tuple(float(value) for value in left_values)
            right = tuple(float(value) for value in right_values)
            bootstrap = paired_model_seed_bootstrap(
                left,
                right,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            comparator_mean = describe(right).mean
            relative = (
                bootstrap.observed_difference / comparator_mean if comparator_mean > 0.0 else None
            )
            metrics.append(
                ContrastMetricResult(
                    metric=metric,
                    method_summary=describe(left),
                    comparator_summary=describe(right),
                    paired_difference=bootstrap,
                    relative_difference=relative,
                )
            )
        results.append(PolicyContrastResult(comparator, tuple(metrics)))
    return tuple(results)


def split_sensitivity(
    records: tuple[FederationResultRecord, ...],
) -> list[dict[str, object]]:
    """Summarize repeated role permutations without treating them as independent subjects."""

    grouped: dict[tuple[int, PolicyId, str], list[float]] = {}
    for row in records:
        for metric in ("mebe", "high_excess", "attack_balanced_macro_tpr"):
            value = getattr(row, metric)
            if value is None:
                continue
            grouped.setdefault((row.model_seed, row.policy, metric), []).append(float(value))
    rows: list[dict[str, object]] = []
    for (model_seed, policy, metric), values in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1].value, item[0][2]),
    ):
        rows.append(
            {
                "model_seed": model_seed,
                "policy": policy.value,
                "metric": metric,
                **split_sensitivity_summary(tuple(values)),
                "calibration_split_count": len(values),
            }
        )
    return rows
