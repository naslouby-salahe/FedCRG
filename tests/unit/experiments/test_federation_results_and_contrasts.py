from __future__ import annotations

from pathlib import Path

import pytest

from fedcrg.config import Study
from fedcrg.evidence.models import RunConfig, RunManifest
from fedcrg.evidence.store import atomic_write_json
from fedcrg.experiments.analyses import (
    FederationResultRecord,
    confirmatory_contrasts,
    load_federation_results,
)
from fedcrg.paths import RunLayout
from fedcrg.thresholding.metrics import FederationMetrics
from fedcrg.types import DatasetId, ExperimentId, ExperimentStatus, PolicyId

_SHA = "a" * 64


@pytest.fixture(scope="module")
def real_config():
    return Study.load().resolve(ExperimentId.PRIMARY_NBAIOT)


def _write_run(
    root: Path,
    run_id: str,
    *,
    real_config,
    status: ExperimentStatus = ExperimentStatus.COMPLETE,
    policy: PolicyId = PolicyId.FEDCRG,
    model_seed: int = 0,
    calibration_seed: int = 2000,
    write_manifest: bool = True,
    write_run_config: bool = True,
    write_federation_metrics: bool = True,
    corrupt_manifest: bool = False,
    mebe: float = 0.01,
    high_excess: float = 0.02,
    band_violation_rate: float = 0.0,
    attack_balanced_macro_tpr: float | None = 0.8,
) -> RunLayout:
    layout = RunLayout(root / run_id)
    if write_manifest:
        if corrupt_manifest:
            layout.manifest.parent.mkdir(parents=True, exist_ok=True)
            layout.manifest.write_text("{not valid json", encoding="utf-8")
        else:
            manifest = RunManifest(
                run_id=run_id,
                experiment_id=ExperimentId.PRIMARY_NBAIOT,
                policy_id=policy,
                config_hash=_SHA,
                model_seed=model_seed,
                calibration_seed=calibration_seed,
                status=status,
            )
            atomic_write_json(layout.manifest, manifest)
    if write_run_config:
        run_config = RunConfig(
            run_id=run_id,
            experiment_id=ExperimentId.PRIMARY_NBAIOT,
            policy_id=policy,
            parameters=real_config,
            model_seed=model_seed,
            calibration_seed=calibration_seed,
            config_hash=_SHA,
            data_spec_hash=_SHA,
            training_spec_hash=_SHA,
            git_commit="deadbeef",
            git_clean=True,
            environment_pin_sha256=_SHA,
            environment_pin_kind="test",
        )
        atomic_write_json(layout.run_config, run_config)
    if write_federation_metrics:
        federation = FederationMetrics(
            policy=policy,
            client_count=9,
            mebe=mebe,
            high_excess=high_excess,
            band_violation_rate=band_violation_rate,
            mafe=0.0,
            max_fpr=0.02,
            fpr_iqr=0.001,
            attack_balanced_macro_tpr=attack_balanced_macro_tpr,
            macro_tpr=0.9,
            worst_client_tpr=0.7,
            worst_client_attack_balanced_tpr=0.6,
            mean_f1=0.85,
        )
        atomic_write_json(layout.federation_metrics, federation)
    return layout


def test_load_federation_results_includes_only_complete_runs_with_all_artifacts(
    tmp_path: Path, real_config
) -> None:
    _write_run(tmp_path, "complete", real_config=real_config, status=ExperimentStatus.COMPLETE)
    _write_run(tmp_path, "pending", real_config=real_config, status=ExperimentStatus.PENDING)
    _write_run(tmp_path, "no-metrics", real_config=real_config, write_federation_metrics=False)
    _write_run(tmp_path, "no-manifest", real_config=real_config, write_manifest=False)
    _write_run(tmp_path, "corrupt", real_config=real_config, corrupt_manifest=True)

    run_dirs = tuple((tmp_path / name) for name in sorted(p.name for p in tmp_path.iterdir()))
    results = load_federation_results(run_dirs)

    assert len(results) == 1
    record = results[0]
    assert isinstance(record, FederationResultRecord)
    assert record.run_id == "complete"
    assert record.dataset_id is DatasetId.NBAIOT
    assert record.policy is PolicyId.FEDCRG
    assert record.mebe == pytest.approx(0.01)


def test_load_federation_results_on_empty_input_is_empty() -> None:
    assert load_federation_results(()) == ()


def _record(
    *,
    model_seed: int,
    policy: PolicyId,
    calibration_seed: int = 2000,
    mebe: float,
    high_excess: float,
    attack_balanced_macro_tpr: float | None,
) -> FederationResultRecord:
    return FederationResultRecord(
        run_id=f"{policy}-{model_seed}",
        experiment_id=ExperimentId.PRIMARY_NBAIOT,
        dataset_id=DatasetId.NBAIOT,
        model_seed=model_seed,
        calibration_seed=calibration_seed,
        policy=policy,
        mebe=mebe,
        high_excess=high_excess,
        band_violation_rate=0.0,
        attack_balanced_macro_tpr=attack_balanced_macro_tpr,
    )


def _fedcrg_and_comparator_records(
    *,
    fedcrg_mebe: tuple[float, float],
    comparator_mebe: tuple[float, float],
) -> tuple[FederationResultRecord, ...]:
    comparators = (
        PolicyId.GLOBAL_QUANTILE,
        PolicyId.LOCAL_QUANTILE,
        PolicyId.READINESS_ONLY,
        PolicyId.SHRINKAGE,
    )
    records = [
        _record(
            model_seed=seed,
            policy=PolicyId.FEDCRG,
            mebe=fedcrg_mebe[index],
            high_excess=0.1,
            attack_balanced_macro_tpr=0.8,
        )
        for index, seed in enumerate((1, 2))
    ]
    for comparator in comparators:
        records.extend(
            _record(
                model_seed=seed,
                policy=comparator,
                mebe=comparator_mebe[index],
                high_excess=0.2,
                attack_balanced_macro_tpr=0.5,
            )
            for index, seed in enumerate((1, 2))
        )
    return tuple(records)


def test_confirmatory_contrasts_computes_paired_differences_for_every_comparator() -> None:
    records = _fedcrg_and_comparator_records(fedcrg_mebe=(0.1, 0.3), comparator_mebe=(0.2, 0.4))
    results = confirmatory_contrasts(
        records,
        named_calibration_seed=2000,
        expected_model_seeds=(1, 2),
        bootstrap_seed=1,
        bootstrap_replicates=200,
        bootstrap_ci_percentiles=(2.5, 97.5),
    )
    assert len(results) == 4
    for contrast in results:
        mebe_metric = next(m for m in contrast.metrics if m.metric == "mebe")
        assert mebe_metric.paired_difference.observed_difference == pytest.approx(-0.1)
        assert mebe_metric.relative_difference is not None
        assert {m.metric for m in contrast.metrics} == {
            "mebe",
            "high_excess",
            "attack_balanced_macro_tpr",
        }


def test_confirmatory_contrasts_skips_relative_difference_when_comparator_mean_is_zero() -> None:
    records = _fedcrg_and_comparator_records(fedcrg_mebe=(0.0, 0.0), comparator_mebe=(0.0, 0.0))
    results = confirmatory_contrasts(
        records,
        named_calibration_seed=2000,
        expected_model_seeds=(1, 2),
        bootstrap_seed=1,
        bootstrap_replicates=50,
        bootstrap_ci_percentiles=(5, 95),
    )
    mebe_metric = next(m for m in results[0].metrics if m.metric == "mebe")
    assert mebe_metric.relative_difference is None


def test_confirmatory_contrasts_skips_metric_when_any_value_is_none() -> None:
    comparators = (
        PolicyId.GLOBAL_QUANTILE,
        PolicyId.LOCAL_QUANTILE,
        PolicyId.READINESS_ONLY,
        PolicyId.SHRINKAGE,
    )
    records = [
        _record(
            model_seed=1,
            policy=PolicyId.FEDCRG,
            mebe=0.1,
            high_excess=0.1,
            attack_balanced_macro_tpr=None,
        ),
        _record(
            model_seed=2,
            policy=PolicyId.FEDCRG,
            mebe=0.2,
            high_excess=0.2,
            attack_balanced_macro_tpr=None,
        ),
    ]
    for comparator in comparators:
        records.append(
            _record(
                model_seed=1,
                policy=comparator,
                mebe=0.15,
                high_excess=0.15,
                attack_balanced_macro_tpr=0.5,
            )
        )
        records.append(
            _record(
                model_seed=2,
                policy=comparator,
                mebe=0.25,
                high_excess=0.25,
                attack_balanced_macro_tpr=0.5,
            )
        )
    results = confirmatory_contrasts(
        tuple(records),
        named_calibration_seed=2000,
        expected_model_seeds=(1, 2),
        bootstrap_seed=1,
        bootstrap_replicates=50,
        bootstrap_ci_percentiles=(5, 95),
    )
    assert {m.metric for m in results[0].metrics} == {"mebe", "high_excess"}


def test_confirmatory_contrasts_rejects_unexpected_fedcrg_model_seeds() -> None:
    records = _fedcrg_and_comparator_records(fedcrg_mebe=(0.1, 0.3), comparator_mebe=(0.2, 0.4))
    with pytest.raises(ValueError):
        confirmatory_contrasts(
            records,
            named_calibration_seed=2000,
            expected_model_seeds=(1, 2, 3),
            bootstrap_seed=1,
            bootstrap_replicates=50,
            bootstrap_ci_percentiles=(5, 95),
        )


def test_confirmatory_contrasts_rejects_missing_comparator_model_seeds() -> None:
    records = tuple(
        r
        for r in _fedcrg_and_comparator_records(fedcrg_mebe=(0.1, 0.3), comparator_mebe=(0.2, 0.4))
        if not (r.policy is PolicyId.GLOBAL_QUANTILE and r.model_seed == 2)
    )
    with pytest.raises(ValueError):
        confirmatory_contrasts(
            records,
            named_calibration_seed=2000,
            expected_model_seeds=(1, 2),
            bootstrap_seed=1,
            bootstrap_replicates=50,
            bootstrap_ci_percentiles=(5, 95),
        )
