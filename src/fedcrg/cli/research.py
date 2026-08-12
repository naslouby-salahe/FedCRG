"""Statistical precomputation, synthetic experiments, robustness, and benchmarking."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import click
import numpy as np

from fedcrg.analysis.benchmark import benchmark
from fedcrg.application.robustness import RunRobustness
from fedcrg.application.synthetic import RunSyntheticExperiments
from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.cli.shared import load_config
from fedcrg.protocol.decision import ThresholdDecisionEngine
from fedcrg.protocol.mismatch import ReferenceMismatchEvaluator
from fedcrg.protocol.readiness import CalibrationReadinessEvaluator, ReadinessPlanCache
from fedcrg.protocol.reference import ReferenceThresholdEstimator


@click.group(name="tables")
def tables_group() -> None:
    """Precompute protocol tables that are independent of observed client scores."""


@tables_group.command(name="precompute-readiness")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("outputs/cache/precomputed/readiness_plans.json"))
def precompute_readiness(config_path: Path, output: Path) -> None:
    config = load_config(config_path)
    cache = ReadinessPlanCache(output)
    counts = sorted({
        config.dataset.split.calibration_benign,
        149, 270, 500, 694, 1000, 1400, 1415, 1416, 1500, 2000, 2435, 2861, 3000, 5970,
    })
    for count in counts:
        cache.precompute(count, config.protocol.band, config.protocol.readiness_assurance)
    click.echo(str(output))


@click.group(name="synthetic")
def synthetic_group() -> None:
    """Run pre-registered synthetic validation cells."""


@synthetic_group.command(name="run")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--experiment", type=click.Choice(["S1", "S2", "S3", "S4", "S5", "S6"]), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def synthetic_run(config_path: Path, experiment: str, output: Path) -> None:
    config = load_config(config_path)
    runner = RunSyntheticExperiments()
    if experiment == "S1":
        path = runner.run_s1(config, output)
    elif experiment == "S2":
        path = runner.run_s2(config, output)
    elif experiment == "S3":
        path = runner.run_s3(config, output)
    elif experiment == "S4":
        path = runner.run_s4(config, output)
    elif experiment == "S5":
        path = runner.run_s5(config, output)
    else:
        path = runner.run_s6(output)
    click.echo(str(path))


@click.group(name="robustness")
def robustness_group() -> None:
    """Run mandatory outcome-independent robustness checks."""


@robustness_group.command(name="second-detector")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--model-seed", type=int, required=True)
def second_detector(config_path: Path, prepared_root: Path, model_seed: int) -> None:
    model, manifest = RunRobustness().train_second_detector(load_config(config_path), prepared_root, model_seed)
    click.echo(f"model={model}\nmanifest={manifest}")


@click.command(name="benchmark")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("outputs/reports/latest/benchmark.json"))
def benchmark_command(config_path: Path, output: Path) -> None:
    config = load_config(config_path)
    rng = np.random.default_rng(123456)
    references = {f"client_{i}": rng.normal(size=config.dataset.split.reference_benign).astype(np.float64) for i in range(max(config.dataset.expected_clients or 9, 1))}
    calibration = rng.normal(size=config.dataset.split.calibration_benign).astype(np.float64)
    mismatch = rng.normal(size=config.dataset.split.mismatch_benign).astype(np.float64)
    reference_estimator = ReferenceThresholdEstimator()
    reference = reference_estimator.estimate(references, config.protocol.alpha)
    readiness_cache = ReadinessPlanCache()
    plan = readiness_cache.precompute(len(calibration), config.protocol.band, config.protocol.readiness_assurance)
    readiness_evaluator = CalibrationReadinessEvaluator()
    mismatch_evaluator = ReferenceMismatchEvaluator()
    decision_engine = ThresholdDecisionEngine()

    def reference_op() -> object: return reference_estimator.estimate(references, config.protocol.alpha)
    def readiness_op() -> object: return readiness_evaluator.evaluate(calibration, plan)
    def mismatch_op() -> object: return mismatch_evaluator.evaluate(mismatch, reference.value, config.protocol.band, config.protocol.mismatch_confidence)
    readiness = readiness_op()
    mismatch_result = mismatch_op()
    def decision_op() -> object: return decision_engine.decide(reference, readiness, mismatch_result)

    rows = [asdict(benchmark(name, operation)) for name, operation in {
        "reference_construction": reference_op,
        "readiness_lookup_and_order_statistic": readiness_op,
        "reference_mismatch_interval": mismatch_op,
        "policy_decision": decision_op,
    }.items()]
    atomic_write_json(output, {"benchmark": "R13", "rows": rows})
    click.echo(str(output))


@click.group(name="sensitivity")
def sensitivity_group() -> None:
    """Run pre-registered real-score sensitivities on a frozen score cache."""


@sensitivity_group.command(name="run")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--score-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--experiment", type=click.Choice(["R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def sensitivity_run(config_path: Path, score_root: Path, experiment: str, output: Path) -> None:
    from fedcrg.application.sensitivity import RunRealSensitivities
    from fedcrg.scoring.cache import ScoreCache

    config = load_config(config_path)
    scores = ScoreCache().load(score_root)
    runner = RunRealSensitivities()
    actions = {
        "R2": runner.readiness_sample_size,
        "R3": runner.mismatch_sample_size,
        "R4": runner.tolerance,
        "R5": runner.target_fpr,
        "R6": runner.assurance,
        "R7": runner.multiplicity,
        "R8": runner.source_order_test_blocks,
        "R9": runner.real_contamination,
    }
    path = actions[experiment](config, scores, output)
    click.echo(str(path))
