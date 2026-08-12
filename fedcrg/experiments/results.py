"""Experiment Results Collection and Analysis.

Provides utilities for collecting, aggregating, and analyzing experiment results.
Includes functions for computing aggregate metrics, generating reports, and
verifying experiment completeness.
"""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from fedcrg.experiments.registry import ExperimentID, ExperimentRegistry, get_registry
from fedcrg.experiments.executor import ExperimentResult
from fedcrg.experiments.synthetic import SyntheticExperimentResult
from fedcrg.experiments.real_data import RealDataExperimentResult


@dataclass
class ResultCollector:
    """Collects and manages experiment results.
    
    Provides functionality to:
    - Load results from disk
    - Aggregate results across experiments
    - Compute summary statistics
    - Generate reports
    - Verify completeness
    """
    
    results_dir: str = "artifacts/experiments"
    results: Dict[str, ExperimentResult] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize by loading existing results."""
        self.registry = get_registry()
        self._load_results()
    
    def _load_results(self):
        """Load all experiment results from the results directory."""
        results_path = Path(self.results_dir)
        if not results_path.exists():
            return
        
        for result_file in results_path.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    result = ExperimentResult.deserialize(f.read())
                self.results[result.experiment_id] = result
            except Exception as e:
                warnings.warn(f"Could not load result from {result_file}: {e}")
    
    def get_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get a specific experiment result."""
        return self.results.get(experiment_id)
    
    def get_results_by_type(
        self,
        experiment_type: str = "all",
    ) -> Dict[str, ExperimentResult]:
        """Get results filtered by type.
        
        Args:
            experiment_type: "all", "synthetic", "real_data", or "confirmatory"
        
        Returns:
            Dict mapping experiment_id to result
        """
        if experiment_type == "all":
            return self.results.copy()
        
        if experiment_type == "synthetic":
            synthetic_ids = self.registry.list_synthetic()
            return {k: v for k, v in self.results.items() if k in synthetic_ids}
        
        if experiment_type == "real_data":
            real_ids = self.registry.list_real_data()
            return {k: v for k, v in self.results.items() if k in real_ids}
        
        if experiment_type == "confirmatory":
            confirmatory_ids = self.registry.list_confirmatory()
            return {k: v for k, v in self.results.items() if k in confirmatory_ids}
        
        return {}
    
    def get_status(self) -> Dict[str, Any]:
        """Get a summary status of all experiments."""
        all_ids = self.registry.list_all()
        
        return {
            "total": len(all_ids),
            "completed": len(self.results),
            "missing": len(all_ids) - len(self.results),
            "synthetic_completed": len(self.get_results_by_type("synthetic")),
            "synthetic_total": len(self.registry.list_synthetic()),
            "real_data_completed": len(self.get_results_by_type("real_data")),
            "real_data_total": len(self.registry.list_real_data()),
            "confirmatory_completed": len(self.get_results_by_type("confirmatory")),
            "confirmatory_total": len(self.registry.list_confirmatory()),
        }
    
    def list_missing(self) -> List[str]:
        """List experiment IDs that have not been run."""
        all_ids = self.registry.list_all()
        return [exp_id for exp_id in all_ids if exp_id not in self.results]
    
    def list_failed(self) -> List[str]:
        """List experiment IDs that failed."""
        return [
            exp_id for exp_id, result in self.results.items()
            if result.status == "FAILED"
        ]
    
    def list_complete(self) -> List[str]:
        """List experiment IDs that completed successfully."""
        return [
            exp_id for exp_id, result in self.results.items()
            if result.status == "COMPLETE"
        ]


@dataclass
class AggregateMetrics:
    """Container for aggregate metrics across experiments."""
    
    # Per-experiment metrics
    experiment_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Aggregate statistics
    summary_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Correlation matrices
    correlations: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_metrics": self.experiment_metrics,
            "summary_stats": {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in self.summary_stats.items()
            },
        }
    
    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "AggregateMetrics":
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            experiment_metrics=data.get("experiment_metrics", {}),
            summary_stats=data.get("summary_stats", {}),
        )


def compute_aggregate_metrics(
    results: Dict[str, ExperimentResult],
) -> AggregateMetrics:
    """Compute aggregate metrics from experiment results.
    
    This function extracts metrics from all completed experiments and
    computes summary statistics.
    
    Args:
        results: Dict mapping experiment_id to ExperimentResult
    
    Returns:
        AggregateMetrics with all computed statistics
    """
    experiment_metrics: Dict[str, Dict[str, float]] = {}
    
    for exp_id, result in results.items():
        if result.status != "COMPLETE":
            continue
        
        # Extract metrics from the result
        # The structure depends on whether it's synthetic or real data
        if hasattr(result.result, "results"):
            # Synthetic or real data result with results dict
            raw_results = result.result.results
            
            # Flatten the results to get numeric values
            metrics = _flatten_results(raw_results)
            experiment_metrics[exp_id] = metrics
        else:
            # Fallback for unexpected structure
            experiment_metrics[exp_id] = {}
    
    # Compute summary statistics
    summary_stats: Dict[str, Dict[str, float]] = {}
    
    # Collect all metric names
    all_metric_names = set()
    for metrics in experiment_metrics.values():
        all_metric_names.update(metrics.keys())
    
    # For each metric, compute statistics across experiments
    for metric_name in all_metric_names:
        values = []
        for exp_id, metrics in experiment_metrics.items():
            if metric_name in metrics:
                values.append(metrics[metric_name])
        
        if len(values) > 0:
            summary_stats[metric_name] = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }
    
    return AggregateMetrics(
        experiment_metrics=experiment_metrics,
        summary_stats=summary_stats,
    )


def _flatten_results(
    results: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, float]:
    """Flatten a nested results dictionary to a flat dict of float values."""
    flat: Dict[str, float] = {}
    
    for key, value in results.items():
        full_key = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively flatten
            nested = _flatten_results(value, prefix=f"{full_key}.")
            flat.update(nested)
        elif isinstance(value, (int, float, np.number)):
            flat[full_key] = float(value)
        elif isinstance(value, list) and len(value) > 0:
            # Take mean of list values
            numeric_values = [float(x) for x in value if isinstance(x, (int, float, np.number))]
            if numeric_values:
                flat[f"{full_key}.mean"] = float(np.mean(numeric_values))
                flat[f"{full_key}.count"] = float(len(numeric_values))
    
    return flat


def serialize_results(
    results: Dict[str, ExperimentResult],
    output_path: str,
) -> None:
    """Serialize all results to a single JSON file.
    
    Args:
        results: Dict mapping experiment_id to ExperimentResult
        output_path: Path to output file
    """
    serialized = {}
    for exp_id, result in results.items():
        serialized[exp_id] = result.to_dict()
    
    with open(output_path, 'w') as f:
        json.dump(serialized, f, indent=2, default=str)


def deserialize_results(
    input_path: str,
) -> Dict[str, ExperimentResult]:
    """Deserialize all results from a JSON file.
    
    Args:
        input_path: Path to input file
    
    Returns:
        Dict mapping experiment_id to ExperimentResult
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    results = {}
    for exp_id, result_data in data.items():
        result = ExperimentResult.from_dict(result_data)
        results[exp_id] = result
    
    return results


def generate_experiment_report(
    results: Dict[str, ExperimentResult],
    output_path: str,
    format: str = "markdown",
) -> None:
    """Generate a human-readable report of experiment results.
    
    Args:
        results: Dict mapping experiment_id to ExperimentResult
        output_path: Path to output file
        format: Output format: "markdown" or "text"
    """
    registry = get_registry()
    
    lines = []
    
    if format == "markdown":
        lines.append("# FedCRG Experiment Report")
        lines.append("")
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append("")
        
        # Status summary
        lines.append("## Status Summary")
        lines.append("")
        status = ResultCollector(results_dir="").get_status()
        lines.append(f"- **Total experiments**: {status['total']}")
        lines.append(f"- **Completed**: {status['completed']}")
        lines.append(f"- **Synthetic completed**: {status['synthetic_completed']}/{status['synthetic_total']}")
        lines.append(f"- **Real data completed**: {status['real_data_completed']}/{status['real_data_total']}")
        lines.append(f"- **Confirmatory completed**: {status['confirmatory_completed']}/{status['confirmatory_total']}")
        lines.append("")
        
        # Completed experiments
        lines.append("## Completed Experiments")
        lines.append("")
        lines.append("| ID | Name | Status | Duration |")
        lines.append("|---|---|---|---|")
        
        for exp_id in sorted(results.keys()):
            result = results[exp_id]
            try:
                exp_config = registry.get(exp_id)
                name = exp_config.name
            except:
                name = "Unknown"
            
            duration = (result.end_time - result.start_time).total_seconds()
            lines.append(f"| {exp_id} | {name} | {result.status} | {duration:.2f}s |")
        
        lines.append("")
        
        # Missing experiments
        collector = ResultCollector(results_dir="")
        collector.results = results
        missing = collector.list_missing()
        if missing:
            lines.append("## Missing Experiments")
            lines.append("")
            lines.append(", ".join(missing))
            lines.append("")
        
        # Failed experiments
        failed = collector.list_failed()
        if failed:
            lines.append("## Failed Experiments")
            lines.append("")
            lines.append(", ".join(failed))
            lines.append("")
    
    else:  # text format
        lines.append("FedCRG Experiment Report")
        lines.append("=" * 40)
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append("")
        
        status = ResultCollector(results_dir="").get_status()
        lines.append(f"Total experiments: {status['total']}")
        lines.append(f"Completed: {status['completed']}")
        lines.append(f"Synthetic: {status['synthetic_completed']}/{status['synthetic_total']}")
        lines.append(f"Real data: {status['real_data_completed']}/{status['real_data_total']}")
        lines.append(f"Confirmatory: {status['confirmatory_completed']}/{status['confirmatory_total']}")
        lines.append("")
        
        lines.append("Completed Experiments:")
        lines.append("-" * 40)
        
        for exp_id in sorted(results.keys()):
            result = results[exp_id]
            try:
                exp_config = registry.get(exp_id)
                name = exp_config.name
            except:
                name = "Unknown"
            
            duration = (result.end_time - result.start_time).total_seconds()
            lines.append(f"  {exp_id}: {name} ({result.status}, {duration:.2f}s)")
    
    # Write output
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def verify_experiment_completeness(
    results: Dict[str, ExperimentResult],
    required_experiments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Verify that all required experiments have been completed.
    
    Args:
        results: Dict mapping experiment_id to ExperimentResult
        required_experiments: Optional list of experiment IDs to check
                           If None, checks all registered experiments
    
    Returns:
        Dict with verification results including:
        - complete: bool indicating if all required experiments are complete
        - missing: list of missing experiment IDs
        - failed: list of failed experiment IDs
        - incomplete: list of incomplete experiment IDs
    """
    registry = get_registry()
    
    if required_experiments is None:
        required_experiments = registry.list_all()
    
    missing = []
    failed = []
    incomplete = []
    
    for exp_id in required_experiments:
        if exp_id not in results:
            missing.append(exp_id)
        else:
            result = results[exp_id]
            if result.status == "FAILED":
                failed.append(exp_id)
            elif result.status != "COMPLETE":
                incomplete.append(exp_id)
    
    return {
        "complete": len(missing) == 0 and len(failed) == 0 and len(incomplete) == 0,
        "missing": missing,
        "failed": failed,
        "incomplete": incomplete,
        "total_required": len(required_experiments),
        "total_completed": len(required_experiments) - len(missing) - len(failed) - len(incomplete),
    }


def generate_verify_report(
    results: Dict[str, ExperimentResult],
    output_path: str,
) -> None:
    """Generate a verification report for `fedcrg verify` command.
    
    Args:
        results: Dict mapping experiment_id to ExperimentResult
        output_path: Path to output file
    """
    verification = verify_experiment_completeness(results)
    
    lines = [
        "# FedCRG Verification Report",
        "",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
        "## Overall Status",
        "",
        f"- **Complete**: {'YES' if verification['complete'] else 'NO'}",
        f"- **Total required**: {verification['total_required']}",
        f"- **Total completed**: {verification['total_completed']}",
        "",
    ]
    
    if verification['missing']:
        lines.append("## Missing Experiments")
        lines.append("")
        for exp_id in verification['missing']:
            lines.append(f"- {exp_id}")
        lines.append("")
    
    if verification['failed']:
        lines.append("## Failed Experiments")
        lines.append("")
        for exp_id in verification['failed']:
            lines.append(f"- {exp_id}")
        lines.append("")
    
    if verification['incomplete']:
        lines.append("## Incomplete Experiments")
        lines.append("")
        for exp_id in verification['incomplete']:
            lines.append(f"- {exp_id}")
        lines.append("")
    
    if verification['complete']:
        lines.append("## All Experiments Verified")
        lines.append("")
        lines.append("All required experiments have been completed successfully.")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
