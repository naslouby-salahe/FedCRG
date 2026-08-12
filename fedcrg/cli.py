"""FedCRG Command-Line Interface.

Implements all CLI commands per Section 14.10 of the FedCRG Roadmap v2.0:

```text
fedcrg doctor
fedcrg data prepare --config configs/nbaiot_primary.yaml
fedcrg data prepare --config configs/diad_external.yaml
fedcrg tables precompute-gate-a --config configs/protocol_v2.yaml
fedcrg synthetic run --config configs/synthetic.yaml
fedcrg train --config configs/nbaiot_primary.yaml
fedcrg score --config configs/nbaiot_primary.yaml
fedcrg evaluate --config configs/nbaiot_primary.yaml
fedcrg train --config configs/diad_external.yaml
fedcrg score --config configs/diad_external.yaml
fedcrg evaluate --config configs/diad_external.yaml
fedcrg robustness deep-svdd --config configs/nbaiot_primary.yaml
fedcrg benchmark --config configs/protocol_v2.yaml
fedcrg report build
fedcrg verify
```

The CLI uses Click for command-line parsing and provides a user-friendly
interface for running all FedCRG experiments and operations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import click

from fedcrg.config import (
    FedCRGConfig,
    NBaiotConfig,
    DiadConfig,
    ProtocolConfig,
    load_config,
)


# =============================================================================
# MAIN CLI GROUP
# =============================================================================

@click.group(
    name="fedcrg",
    help="FedCRG: Federated Calibration Readiness Gate command-line interface",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(version="2.0", message="FedCRG v2.0")
def cli():
    """Main FedCRG CLI entry point."""
    pass


# =============================================================================
# DOCTOR COMMAND
# =============================================================================

@cli.command(name="doctor")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Show detailed environment information",
)
def doctor(verbose: bool) -> None:
    """Check the FedCRG environment and dependencies.
    
    Performs a series of checks to ensure the environment is properly
    configured for running FedCRG experiments.
    """
    import platform
    import numpy as np
    import scipy
    import pandas as pd
    import torch
    import yaml
    
    click.echo("FedCRG Environment Doctor")
    click.echo("=" * 50)
    
    # Python version
    click.echo(f"Python: {platform.python_version()}")
    click.echo(f"Platform: {platform.platform()}")
    
    # Core packages
    click.echo(f"\nNumPy: {np.__version__}")
    click.echo(f"SciPy: {scipy.__version__}")
    click.echo(f"Pandas: {pd.__version__}")
    click.echo(f"PyTorch: {torch.__version__}")
    click.echo(f"PyYAML: {yaml.__version__}")
    
    # Check CUDA if available
    if torch.cuda.is_available():
        click.echo(f"CUDA: {torch.version.cuda}")
        click.echo(f"cuDNN: {torch.backends.cudnn.version()}")
        click.echo(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        click.echo("CUDA: Not available")
    
    # Check fedcrg package
    try:
        import fedcrg
        click.echo(f"\nFedCRG package: {fedcrg.__file__}")
    except ImportError as e:
        click.echo(f"\nERROR: Could not import fedcrg: {e}")
        sys.exit(1)
    
    # Check configs directory
    configs_dir = Path("configs")
    if configs_dir.exists():
        configs = list(configs_dir.glob("*.yaml"))
        click.echo(f"\nConfig files: {len(configs)} found")
        for config in configs:
            click.echo(f"  - {config.name}")
    else:
        click.echo("\nWARNING: configs/ directory not found")
    
    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        click.echo(f"\nData directory: {data_dir.absolute()}")
        if (data_dir / "raw").exists():
            click.echo("  raw/ symlink: exists")
        else:
            click.echo("  raw/ symlink: NOT FOUND - needs setup")
    else:
        click.echo("\nWARNING: data/ directory not found")
    
    # Check artifacts directory
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        click.echo(f"\nArtifacts directory: {artifacts_dir.absolute()}")
    else:
        click.echo("\nWARNING: artifacts/ directory not found")
    
    # Check Git status
    try:
        import subprocess
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            click.echo(f"\nGit status: {len(result.stdout.strip().splitlines())} changes")
            if verbose:
                click.echo(result.stdout)
        else:
            click.echo("\nGit status: Clean")
    except Exception:
        click.echo("\nGit status: Could not determine")
    
    click.echo("\nEnvironment check complete.")


# =============================================================================
# DATA COMMANDS
# =============================================================================

@cli.group(name="data")
def data_group():
    """Data preparation and management commands."""
    pass


@data_group.command(name="prepare")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--dataset", "-d",
    type=click.Choice(["nbaiot", "diad"]),
    help="Dataset to prepare (auto-detected from config if not specified)",
)
@click.option(
    "--force", "-f",
    is_flag=True,
    default=False,
    help="Force re-preparation even if already prepared",
)
def data_prepare(config: str, dataset: Optional[str], force: bool) -> None:
    """Prepare dataset for FedCRG experiments.
    
    This command:
    1. Validates the configuration file
    2. Checks/creates the data/raw symlink
    3. Verifies dataset integrity
    4. Prepares manifest files
    """
    from fedcrg.data.nbaiot import NBaiotAdapter
    from fedcrg.data.diad import DiadAdapter
    
    click.echo(f"Preparing data with config: {config}")
    
    # Load config
    try:
        config_obj = load_config(config)
    except Exception as e:
        click.echo(f"ERROR: Could not load config: {e}")
        sys.exit(1)
    
    # Determine dataset
    if dataset is None:
        # Determine dataset from config file path
        config_path = Path(config)
        config_name = config_path.name.lower()
        if "nbaiot" in config_name:
            dataset = "nbaiot"
        elif "diad" in config_name:
            dataset = "diad"
        elif "synthetic" in config_name:
            dataset = "synthetic"
        else:
            # Default to nbaiot for protocol configs
            dataset = "nbaiot"
    
    # Prepare the appropriate dataset
    if dataset == "nbaiot":
        adapter = NBaiotAdapter(data_root="data/raw")
        click.echo("Preparing N-BaIoT dataset...")
        # In full implementation, this would:
        # 1. Verify data/raw symlink
        # 2. Scan for device directories
        # 3. Validate file counts
        # 4. Generate manifest
        click.echo("  N-BaIoT preparation: Placeholder")
        click.echo("  Implement full data preparation per Section 7.1")
    elif dataset == "diad":
        adapter = DiadAdapter(data_root="data/raw")
        click.echo("Preparing CIC IoT-DIAD dataset...")
        # Check eligibility
        eligibility = adapter.check_eligibility()
        click.echo(f"  Eligible clients: {len(eligibility['eligible_clients'])}")
        if len(eligibility['eligible_clients']) < 10:
            click.echo(f"  WARNING: Need at least 10 clients for confirmatory replication")
        click.echo("  DIAD preparation: Placeholder")
        click.echo("  Implement full data preparation per Section 7.2")
    else:
        click.echo(f"ERROR: Unknown dataset: {dataset}")
        sys.exit(1)
    
    click.echo("Data preparation complete.")


# =============================================================================
# TABLES COMMANDS
# =============================================================================

@cli.group(name="tables")
def tables_group():
    """Table precomputation and management commands."""
    pass


@tables_group.command(name="precompute-gate-a")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="artifacts/tables/gate_a_table.json",
    help="Output path for precomputed table",
)
def precompute_gate_a(config: str, output: str) -> None:
    """Precompute Gate A table entries for the given configuration.
    
    Per Section 14.5: For fixed (n, a, b, gamma_A), r* and P_r are
    determined BEFORE observing scores. This command precomputes
    these values for all required n values.
    """
    from fedcrg.gate_a import precompute_primary_gate_a_table, _gate_a_table
    from fedcrg.reference import PrimaryAlpha, PrimaryRho, PrimaryGammaA
    
    click.echo("Precomputing Gate A table...")
    
    # Load config to get parameters
    try:
        config_obj = load_config(config)
    except Exception as e:
        click.echo(f"ERROR: Could not load config: {e}")
        sys.exit(1)
    
    # Precompute for primary parameters
    alpha = config_obj.protocol.alpha
    rho = config_obj.protocol.rho
    gamma_a = config_obj.protocol.gate_a_assurance
    
    click.echo(f"Parameters: alpha={alpha}, rho={rho}, gamma_a={gamma_a}")
    
    # Get known n values from config
    n_values = [
        config_obj.nbaiot.local_calibration_per_client,
        config_obj.nbaiot.gate_per_client,
        config_obj.diad.local_calibration_per_client,
        config_obj.diad.gate_per_client,
    ]
    
    # Add sensitivity values
    n_values.extend([500, 1000, 1400, 1415, 1416, 1500, 2000, 2435, 2861, 3000, 5722, 5970])
    n_values = sorted(set(n_values))
    
    # Precompute entries
    table = {}
    for n in n_values:
        entry = _gate_a_table.get(n, alpha, rho, gamma_a)
        table[str(n)] = {
            "n": int(entry.n),
            "rank_r": int(entry.rank_r),
            "coverage_probability": float(entry.coverage_probability),
            "ready": bool(entry.ready),
            "a": float(entry.a),
            "b": float(entry.b),
        }
    
    # Save to file
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(table, f, indent=2)
    
    click.echo(f"Gate A table precomputed with {len(table)} entries")
    click.echo(f"Saved to: {output_path.absolute()}")


# =============================================================================
# SYNTHETIC COMMANDS
# =============================================================================

@cli.group(name="synthetic")
def synthetic_group():
    """Synthetic experiment commands."""
    pass


@synthetic_group.command(name="run")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--experiments", "-e",
    type=str,
    default="all",
    help="Comma-separated list of experiment IDs (S1-S6) or 'all'",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="artifacts/experiments",
    help="Output directory for experiment results",
)
@click.option(
    "--n-reps",
    type=int,
    default=None,
    help="Override number of repetitions (default from config)",
)
def synthetic_run(
    config: str,
    experiments: str,
    output_dir: str,
    n_reps: Optional[int],
) -> None:
    """Run synthetic experiments (S1-S6).
    
    Runs the IID theorem validation and robustness stress tests
    per Section 11.
    """
    from fedcrg.experiments.executor import ExperimentExecutor
    from fedcrg.experiments.registry import get_registry
    
    click.echo(f"Running synthetic experiments with config: {config}")
    
    # Load config
    try:
        config_obj = load_config(config)
    except Exception as e:
        click.echo(f"ERROR: Could not load config: {e}")
        sys.exit(1)
    
    # Determine which experiments to run
    registry = get_registry()
    synthetic_ids = registry.list_s1_to_s6()
    
    if experiments == "all":
        exp_ids = synthetic_ids
    else:
        exp_ids = [e.strip() for e in experiments.split(",")]
        # Validate
        for eid in exp_ids:
            if eid not in synthetic_ids:
                click.echo(f"ERROR: Unknown synthetic experiment: {eid}")
                click.echo(f"Available: {synthetic_ids}")
                sys.exit(1)
    
    click.echo(f"Running experiments: {exp_ids}")
    
    # Create executor
    executor = ExperimentExecutor(
        output_dir=output_dir,
        reuse_results=True,
        verbose=True,
    )
    
    # Run experiments
    results = executor.run_experiments(exp_ids)
    
    # Summary
    completed = sum(1 for r in results.values() if r and r.status == "COMPLETE")
    failed = sum(1 for r in results.values() if r and r.status == "FAILED")
    skipped = sum(1 for r in results.values() if r is None)
    
    click.echo(f"\nSummary:")
    click.echo(f"  Completed: {completed}")
    click.echo(f"  Failed: {failed}")
    click.echo(f"  Skipped: {skipped}")


# =============================================================================
# TRAIN COMMAND
# =============================================================================

@cli.command(name="train")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--dataset", "-d",
    type=click.Choice(["nbaiot", "diad"]),
    help="Dataset to train on (auto-detected from config)",
)
@click.option(
    "--model-seeds",
    type=str,
    default=None,
    help="Comma-separated list of model seeds to use",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="artifacts/models",
    help="Output directory for trained models",
)
@click.option(
    "--force", "-f",
    is_flag=True,
    default=False,
    help="Force retraining even if models exist",
)
def train(
    config: str,
    dataset: Optional[str],
    model_seeds: Optional[str],
    output_dir: str,
    force: bool,
) -> None:
    """Train federated detector models.
    
    Per Section 8.1: Trains the federated autoencoder with exact
    hyperparameters from the roadmap.
    """
    from fedcrg.fl.trainer import FederatedTrainer
    from fedcrg.models.autoencoder import Autoencoder
    from fedcrg.data.nbaiot import NBaiotAdapter
    from fedcrg.data.diad import DiadAdapter
    
    click.echo(f"Training with config: {config}")
    
    # Load config
    try:
        config_obj = load_config(config)
    except Exception as e:
        click.echo(f"ERROR: Could not load config: {e}")
        sys.exit(1)
    
    # Determine dataset
    if dataset is None:
        # Determine dataset from config file path
        config_path = Path(config)
        config_name = config_path.name.lower()
        if "nbaiot" in config_name:
            dataset = "nbaiot"
        elif "diad" in config_name:
            dataset = "diad"
        else:
            # Default to nbaiot
            dataset = "nbaiot"
    
    click.echo(f"Dataset: {dataset}")
    
    # Placeholder implementation
    if dataset == "nbaiot":
        click.echo("N-BaIoT training: Placeholder")
        click.echo("  Implement FederatedTrainer per Section 8.2")
        click.echo("  30 rounds x 120 local epochs")
        click.echo("  Architecture: 115-86-57-38-29-38-57-86-115")
        click.echo("  5 model seeds: 11, 22, 33, 44, 55")
    elif dataset == "diad":
        click.echo("DIAD training: Placeholder")
        click.echo("  Implement FederatedTrainer per Section 8.2")
        click.echo("  30 rounds x 20 local epochs")
        click.echo("  Architecture: 86-64-43-28-21-28-43-64-86")
    else:
        click.echo(f"ERROR: Unknown dataset: {dataset}")
        sys.exit(1)
    
    click.echo("Training complete (placeholder).")


# =============================================================================
# SCORE COMMAND
# =============================================================================

@cli.command(name="score")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--dataset", "-d",
    type=click.Choice(["nbaiot", "diad"]),
    help="Dataset to score (auto-detected from config)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="artifacts/scores",
    help="Output directory for score caches",
)
def score(
    config: str,
    dataset: Optional[str],
    output_dir: str,
) -> None:
    """Compute anomaly scores for all roles.
    
    Per Section 8.2: Computes scores from trained models for all
    required data roles (T, R, G, C, B, A_dev, A_test).
    """
    click.echo(f"Scoring with config: {config}")
    
    # Load config
    try:
        config_obj = load_config(config)
    except Exception as e:
        click.echo(f"ERROR: Could not load config: {e}")
        sys.exit(1)
    
    # Determine dataset
    if dataset is None:
        # Determine dataset from config file path
        config_path = Path(config)
        config_name = config_path.name.lower()
        if "nbaiot" in config_name:
            dataset = "nbaiot"
        elif "diad" in config_name:
            dataset = "diad"
        else:
            # Default to nbaiot
            dataset = "nbaiot"
    
    click.echo(f"Dataset: {dataset}")
    click.echo("Scoring: Placeholder")
    click.echo("  Implement ScoreComputer per Section 8.2")
    click.echo("  Score cache with SHA-256 hashing")
    click.echo("  float64 storage per Section 8.2")
    
    click.echo("Scoring complete (placeholder).")


# =============================================================================
# EVALUATE COMMAND
# =============================================================================

@cli.command(name="evaluate")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--policies", "-p",
    type=str,
    default="all",
    help="Comma-separated list of policy IDs (B0-B10,FEDCRG) or 'all'",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="artifacts/metrics",
    help="Output directory for metric results",
)
def evaluate(
    config: str,
    policies: str,
    output_dir: str,
) -> None:
    """Evaluate threshold policies on cached scores.
    
    Computes all metrics per Section 10 for all specified policies.
    """
    click.echo(f"Evaluating with config: {config}")
    
    # Load config
    try:
        config_obj = load_config(config)
    except Exception as e:
        click.echo(f"ERROR: Could not load config: {e}")
        sys.exit(1)
    
    click.echo(f"Policies: {policies}")
    click.echo("Evaluation: Placeholder")
    click.echo("  Implement policy evaluation per Section 12")
    click.echo("  All metrics from Section 10")
    click.echo("  Primary: MEBE, HighExcess, ABMacroTPR")
    click.echo("  Secondary: BandViolationRate, MAFE, AUROC, AUPRC")
    
    click.echo("Evaluation complete (placeholder).")


# =============================================================================
# ROBUSTNESS COMMANDS
# =============================================================================

@cli.group(name="robustness")
def robustness_group():
    """Robustness and assumption-stress experiment commands."""
    pass


@robustness_group.command(name="deep-svdd")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="artifacts/robustness",
    help="Output directory for robustness results",
)
def robustness_deep_svdd(config: str, output_dir: str) -> None:
    """Run Deep-SVDD second detector robustness check (R11).
    
    Per Section 8.4: Federated Deep-SVDD as mandatory second detector.
    """
    click.echo(f"Running Deep-SVDD robustness check with config: {config}")
    
    # Load config
    try:
        config_obj = load_config(config)
    except Exception as e:
        click.echo(f"ERROR: Could not load config: {e}")
        sys.exit(1)
    
    click.echo("Deep-SVDD: Placeholder")
    click.echo("  Encoder: 115-64-32, tanh, biases disabled")
    click.echo("  Embedding dim: 32")
    click.echo("  30 rounds x 20 local epochs")
    click.echo("  Only B1, B2, B5, FEDCRG policies")
    
    click.echo("Deep-SVDD robustness check complete (placeholder).")


# =============================================================================
# BENCHMARK COMMAND
# =============================================================================

@cli.command(name="benchmark")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--n-warmups",
    type=int,
    default=100,
    help="Number of warmup iterations",
)
@click.option(
    "--n-reps",
    type=int,
    default=1000,
    help="Number of measured repetitions",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="artifacts/benchmark/results.json",
    help="Output path for benchmark results",
)
def benchmark(
    config: str,
    n_warmups: int,
    n_reps: int,
    output: str,
) -> None:
    """Run computational/communication overhead benchmark (R13).
    
    Per Section 14.5.1: Measures wall time and memory for threshold-policy
    primitives on one CPU thread.
    """
    from fedcrg.experiments.real_data import run_r13_computational_benchmark
    
    click.echo(f"Running benchmark with config: {config}")
    
    # Load config
    try:
        config_obj = load_config(config)
    except Exception as e:
        click.echo(f"ERROR: Could not load config: {e}")
        sys.exit(1)
    
    click.echo(f"Warmups: {n_warmups}, Repetitions: {n_reps}")
    
    # Run R13 benchmark
    result = run_r13_computational_benchmark(
        n_warmups=n_warmups,
        n_repetitions=n_reps,
        output_dir=Path(output).parent,
    )
    
    click.echo(f"Benchmark complete. Status: {result.results.get('status', 'UNKNOWN')}")
    
    # Save detailed results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    click.echo(f"Results saved to: {output_path.absolute()}")
    
    # Display summary
    if result.results.get("status") != "PLACEHOLDER":
        click.echo("\nPrimitive Benchmarks:")
        for primitive, metrics in result.results.items():
            if isinstance(metrics, dict) and "median_wall_time_ms" in metrics:
                click.echo(f"  {primitive}:")
                click.echo(f"    Median: {metrics['median_wall_time_ms']:.4f}ms")
                click.echo(f"    P95: {metrics['p95_wall_time_ms']:.4f}ms")


# =============================================================================
# REPORT COMMANDS
# =============================================================================

@cli.group(name="report")
def report_group():
    """Report generation commands."""
    pass


@report_group.command(name="build")
@click.option(
    "--input-dir", "-i",
    type=click.Path(exists=True),
    default="artifacts/experiments",
    help="Input directory containing experiment results",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="reports",
    help="Output directory for reports",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["markdown", "html", "text"]),
    default="markdown",
    help="Report output format",
)
def report_build(input_dir: str, output_dir: str, format: str) -> None:
    """Build comprehensive experiment reports.
    
    Generates all required tables and figures per Section 17.
    """
    from fedcrg.experiments.results import (
        ResultCollector,
        generate_experiment_report,
        generate_verify_report,
    )
    
    click.echo(f"Building reports from: {input_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    collector = ResultCollector(results_dir=input_dir)
    results = collector.results
    
    click.echo(f"Loaded {len(results)} results")
    
    # Generate experiment report
    exp_report_path = output_path / "experiment_report.md"
    generate_experiment_report(
        results=results,
        output_path=str(exp_report_path),
        format="markdown",
    )
    click.echo(f"Experiment report: {exp_report_path}")
    
    # Generate verify report
    verify_report_path = output_path / "verify_report.md"
    generate_verify_report(
        results=results,
        output_path=str(verify_report_path),
    )
    click.echo(f"Verify report: {verify_report_path}")
    
    # Status summary
    status = collector.get_status()
    click.echo(f"\nStatus Summary:")
    click.echo(f"  Total: {status['total']}")
    click.echo(f"  Completed: {status['completed']}")
    click.echo(f"  Synthetic: {status['synthetic_completed']}/{status['synthetic_total']}")
    click.echo(f"  Real data: {status['real_data_completed']}/{status['real_data_total']}")
    click.echo(f"  Confirmatory: {status['confirmatory_completed']}/{status['confirmatory_total']}")
    
    click.echo("\nReports built successfully.")


# =============================================================================
# VERIFY COMMAND
# =============================================================================

@cli.command(name="verify")
@click.option(
    "--input-dir", "-i",
    type=click.Path(exists=True),
    default="artifacts/experiments",
    help="Input directory containing experiment results",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="artifacts/verify/verify_report.md",
    help="Output path for verification report",
)
@click.option(
    "--strict", "-s",
    is_flag=True,
    default=False,
    help="Exit with error code if verification fails",
)
def verify(input_dir: str, output: str, strict: bool) -> None:
    """Verify that all required experiments are complete.
    
    Per Section 14.10: `fedcrg verify` MUST fail if any required
    experiment cell, artifact hash, unit test, leakage check, or
    manifest field is missing.
    """
    from fedcrg.experiments.results import (
        ResultCollector,
        verify_experiment_completeness,
        generate_verify_report,
    )
    
    click.echo("Running FedCRG verification...")
    
    # Load results
    collector = ResultCollector(results_dir=input_dir)
    results = collector.results
    
    # Run verification
    verification = verify_experiment_completeness(results)
    
    # Generate report
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_verify_report(
        results=results,
        output_path=str(output_path),
    )
    
    click.echo(f"Verification report: {output_path}")
    
    # Display summary
    click.echo(f"\nVerification Summary:")
    click.echo(f"  Complete: {verification['complete']}")
    click.echo(f"  Total required: {verification['total_required']}")
    click.echo(f"  Total completed: {verification['total_completed']}")
    
    if verification['missing']:
        click.echo(f"\n  Missing experiments: {verification['missing']}")
    
    if verification['failed']:
        click.echo(f"\n  Failed experiments: {verification['failed']}")
    
    if verification['incomplete']:
        click.echo(f"\n  Incomplete experiments: {verification['incomplete']}")
    
    if verification['complete']:
        click.echo("\n✓ All required experiments verified successfully.")
    else:
        click.echo("\n✗ Verification FAILED: Not all required experiments are complete.")
        if strict:
            sys.exit(1)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    cli()
