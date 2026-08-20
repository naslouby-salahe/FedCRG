"""Shared fixtures for writing completed run artifacts used by reporting tests."""

from __future__ import annotations

from pathlib import Path

from fedcrg.config import ExperimentConfig
from fedcrg.evidence.models import MetricRecord, RunConfig, RunManifest
from fedcrg.evidence.store import atomic_write_json, write_jsonl
from fedcrg.paths import OutputsLayout
from fedcrg.thresholding.metrics import FederationMetrics
from fedcrg.types import (
    CalibrationSeed,
    ExperimentId,
    ExperimentStatus,
    ModelSeed,
    PolicyId,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_E = "e" * 64


def write_completed_run(
    outputs_root: Path,
    *,
    run_id: str,
    experiment_id: ExperimentId,
    policy_id: PolicyId,
    config: ExperimentConfig | None,
    model_seed: ModelSeed,
    calibration_seed: CalibrationSeed,
    mebe: float = 0.02,
    high_excess: float = 0.01,
    band_violation_rate: float = 0.05,
    mafe: float = 0.03,
    max_fpr: float = 0.1,
    fpr_iqr: float = 0.01,
    attack_balanced_macro_tpr: float | None = 0.9,
    macro_tpr: float = 0.85,
    worst_client_tpr: float = 0.7,
    worst_client_attack_balanced_tpr: float = 0.65,
    mean_f1: float = 0.8,
    include_run_config: bool = True,
    include_federation_metrics: bool = True,
) -> Path:
    layout = OutputsLayout(outputs_root).run(run_id)
    layout.create()
    manifest = RunManifest(
        run_id=run_id,
        experiment_id=experiment_id,
        policy_id=policy_id,
        config_hash=config.config_hash if config is not None else _HASH_A,
        model_seed=model_seed,
        calibration_seed=calibration_seed,
        status=ExperimentStatus.COMPLETE,
    )
    atomic_write_json(layout.manifest, manifest.model_dump(mode="json"))
    if include_run_config:
        run_config = RunConfig(
            run_id=run_id,
            experiment_id=experiment_id,
            policy_id=policy_id,
            parameters=config,
            model_seed=model_seed,
            calibration_seed=calibration_seed,
            config_hash=config.config_hash if config is not None else _HASH_A,
            data_spec_hash=_HASH_B,
            training_spec_hash=_HASH_C,
            git_commit="deadbeef",
            git_clean=True,
            git_patch_sha256=None,
            environment_pin_sha256=_HASH_E,
            environment_pin_kind="conda",
        )
        atomic_write_json(layout.run_config, run_config.model_dump(mode="json"))
    if include_federation_metrics:
        federation = FederationMetrics(
            policy=policy_id,
            client_count=9,
            mebe=mebe,
            high_excess=high_excess,
            band_violation_rate=band_violation_rate,
            mafe=mafe,
            max_fpr=max_fpr,
            fpr_iqr=fpr_iqr,
            attack_balanced_macro_tpr=attack_balanced_macro_tpr,
            macro_tpr=macro_tpr,
            worst_client_tpr=worst_client_tpr,
            worst_client_attack_balanced_tpr=worst_client_attack_balanced_tpr,
            mean_f1=mean_f1,
        )
        atomic_write_json(layout.federation_metrics, federation.model_dump(mode="json"))
        write_jsonl(
            layout.metric_records,
            (
                MetricRecord(
                    run_id=run_id,
                    policy_id=policy_id,
                    client_id="nb01",
                    benign_n=10,
                    attack_n=5,
                    fp=1,
                    tn=9,
                    tp=4,
                    fn=1,
                    fpr=0.1,
                    tpr=0.8,
                    precision=0.8,
                    f1=0.8,
                    balanced_accuracy=0.85,
                    auroc=0.9,
                    auprc=0.85,
                    band_error=0.01,
                    attack_balanced_tpr=0.8,
                ),
            ),
        )
    return layout.root
