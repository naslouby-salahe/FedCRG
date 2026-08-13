"""Statistical precomputation, synthetic experiments, robustness, and benchmarking."""

from __future__ import annotations

from pathlib import Path

import click

from fedcrg.application.benchmark import RunBenchmark
from fedcrg.application.precompute import ProtocolTablePrecomputer
from fedcrg.application.robustness import RunRobustness
from fedcrg.application.synthetic import RunSyntheticExperiments
from fedcrg.cli.shared import load_config


@click.group(name="tables")
def tables_group() -> None:
    """Precompute protocol tables that are independent of observed client scores."""


@tables_group.command(name="precompute-readiness")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("outputs/cache/precomputed/readiness_plans.json"),
)
def precompute_readiness(config_path: Path, output: Path) -> None:
    config = load_config(config_path)
    readiness_path, _mismatch_path = ProtocolTablePrecomputer().precompute(config, output.parent)
    click.echo(str(readiness_path))


@click.group(name="synthetic")
def synthetic_group() -> None:
    """Run pre-registered synthetic validation cells."""


@synthetic_group.command(name="run")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--experiment", type=click.Choice(["S1", "S2", "S3", "S4", "S5", "S6"]), required=True
)
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


@robustness_group.command(name="deep-svdd")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--model-seed", type=int, required=True)
def train_deep_svdd(config_path: Path, prepared_root: Path, model_seed: int) -> None:
    """Train the mandatory outcome-independent Deep-SVDD second score generator."""
    model, manifest = RunRobustness().train_second_detector(
        load_config(config_path), prepared_root, model_seed
    )
    click.echo(f"model={model}\nmanifest={manifest}")


@click.command(name="benchmark")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("outputs/reports/latest/benchmark.json"),
)
def benchmark_command(config_path: Path, output: Path) -> None:
    config = load_config(config_path)
    path = RunBenchmark().run_on_synthetic_evidence(config, output)
    click.echo(str(path))


@click.group(name="sensitivity")
def sensitivity_group() -> None:
    """Run pre-registered real-score sensitivities on a frozen score cache."""


_MODEL_SEED_SENSITIVITIES = frozenset({"R2", "R3", "R4", "R5", "R6"})


@sensitivity_group.command(name="run")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--score-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--experiment",
    type=click.Choice(["R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R12"]),
    required=True,
)
@click.option("--model-seed", type=int, default=None)
@click.option("--calibration-seed", type=int, default=None)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def sensitivity_run(
    config_path: Path,
    score_root: Path,
    prepared_root: Path,
    experiment: str,
    model_seed: int | None,
    calibration_seed: int | None,
    output: Path,
) -> None:
    """Run one pre-registered R2-R9/R12 real-score sensitivity on a frozen score cache."""
    from fedcrg.application.sensitivity import RunRealSensitivities
    from fedcrg.application.source_order import RunSourceOrderCalibration

    config = load_config(config_path)

    if experiment == "R12":
        path = RunSourceOrderCalibration().run(config, prepared_root, score_root, output)
        click.echo(str(path))
        return

    runner = RunRealSensitivities()
    methods = {
        "R2": runner.run_r2,
        "R3": runner.run_r3,
        "R4": runner.run_r4,
        "R5": runner.run_r5,
        "R6": runner.run_r6,
        "R7": runner.run_r7,
        "R8": runner.run_r8,
        "R9": runner.run_r9,
    }
    method = methods[experiment]
    if experiment in _MODEL_SEED_SENSITIVITIES:
        if model_seed is None:
            raise click.UsageError(f"{experiment} requires --model-seed")
        path = method(config, score_root, prepared_root, model_seed, output, calibration_seed)
    else:
        path = method(config, score_root, prepared_root, output, calibration_seed)
    click.echo(str(path))
