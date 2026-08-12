"""Experiment Executor.

Provides the execution engine for running FedCRG experiments.
Handles experiment dependency resolution, score cache management,
and result collection.
"""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import warnings

from fedcrg.experiments.registry import ExperimentID, ExperimentRegistry, get_registry
from fedcrg.experiments.synthetic import (
    SyntheticExperimentResult,
    run_s1_gate_a_theorem,
    run_s2_target_fpr_sensitivity,
    run_s3_temporal_dependence,
    run_s4_calibration_shift,
    run_s5_contamination,
    run_s6_gate_b_power,
)
from fedcrg.experiments.real_data import (
    RealDataExperimentResult,
    run_r1_primary,
    run_r2_gate_a_sweep,
    run_r3_gate_b_sweep,
    run_r4_tolerance_sensitivity,
    run_r5_target_fpr_sensitivity,
    run_r6_assurance_sensitivity,
    run_r7_multiplicity_sensitivity,
    run_r8_source_order,
    run_r9_real_contamination,
    run_r10_diad_replication,
    run_r11_second_detector,
    run_r12_source_order_roles,
    run_r13_computational_benchmark,
    run_r14_diad_feature_sensitivity,
)


@dataclass
class ExperimentResult:
    """Generic experiment result container.
    
    Can hold either synthetic or real data experiment results.
    """
    experiment_id: str
    result: Any  # Either SyntheticExperimentResult or RealDataExperimentResult
    config: Dict[str, Any]
    start_time: datetime
    end_time: datetime
    status: str  # "COMPLETE", "FAILED", "SKIPPED"
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "result": self.result.to_dict() if hasattr(self.result, "to_dict") else self.result,
            "config": self.config,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "status": self.status,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        result = data.get("result", {})
        if isinstance(result, dict) and "experiment_id" in result:
            if result.get("score_cache_hash") is not None:
                result = RealDataExperimentResult.from_dict(result)
            else:
                result = SyntheticExperimentResult.from_dict(result)
        
        return cls(
            experiment_id=data["experiment_id"],
            result=result,
            config=data.get("config", {}),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            status=data["status"],
            error=data.get("error"),
        )
    
    def serialize(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def deserialize(cls, json_str: str) -> "ExperimentResult":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# Map experiment IDs to their runner functions
_EXPERIMENT_RUNNERS: Dict[str, Callable] = {
    "S1": run_s1_gate_a_theorem,
    "S2": run_s2_target_fpr_sensitivity,
    "S3": run_s3_temporal_dependence,
    "S4": run_s4_calibration_shift,
    "S5": run_s5_contamination,
    "S6": run_s6_gate_b_power,
    "R1": run_r1_primary,
    "R2": run_r2_gate_a_sweep,
    "R3": run_r3_gate_b_sweep,
    "R4": run_r4_tolerance_sensitivity,
    "R5": run_r5_target_fpr_sensitivity,
    "R6": run_r6_assurance_sensitivity,
    "R7": run_r7_multiplicity_sensitivity,
    "R8": run_r8_source_order,
    "R9": run_r9_real_contamination,
    "R10": run_r10_diad_replication,
    "R11": run_r11_second_detector,
    "R12": run_r12_source_order_roles,
    "R13": run_r13_computational_benchmark,
    "R14": run_r14_diad_feature_sensitivity,
}


class ExperimentExecutor:
    """Executes FedCRG experiments with dependency management.
    
    This class manages the execution of experiments, including:
    - Resolving experiment dependencies
    - Managing score caches for real data experiments
    - Collecting and serializing results
    - Tracking execution progress
    """
    
    def __init__(
        self,
        output_dir: str = "artifacts/experiments",
        reuse_results: bool = True,
        verbose: bool = False,
    ):
        """Initialize the executor.
        
        Args:
            output_dir: Directory to store experiment results
            reuse_results: Whether to reuse existing results if available
            verbose: Whether to print verbose execution information
        """
        self.output_dir = Path(output_dir)
        self.reuse_results = reuse_results
        self.verbose = verbose
        self.registry = get_registry()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track completed experiments
        self.completed: Dict[str, ExperimentResult] = {}
        self.failed: Dict[str, str] = {}
    
    def _get_result_path(self, experiment_id: str) -> Path:
        """Get the path for storing an experiment's result."""
        return self.output_dir / f"{experiment_id}.json"
    
    def _load_existing_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Load an existing result if it exists."""
        if not self.reuse_results:
            return None
        
        result_path = self._get_result_path(experiment_id)
        if not result_path.exists():
            return None
        
        try:
            with open(result_path, 'r') as f:
                result = ExperimentResult.deserialize(f.read())
            if self.verbose:
                print(f"Reusing existing result for {experiment_id}")
            return result
        except Exception as e:
            if self.verbose:
                print(f"Could not load existing result for {experiment_id}: {e}")
            return None
    
    def _save_result(self, result: ExperimentResult) -> None:
        """Save an experiment result to disk."""
        result_path = self._get_result_path(result.experiment_id)
        
        try:
            with open(result_path, 'w') as f:
                f.write(result.serialize())
            if self.verbose:
                print(f"Saved result for {result.experiment_id} to {result_path}")
        except Exception as e:
            warnings.warn(f"Could not save result for {result.experiment_id}: {e}")
    
    def get_dependencies(self, experiment_id: str) -> List[str]:
        """Get list of experiment IDs that must complete before the given experiment."""
        return self.registry.get_dependencies(experiment_id)
    
    def check_dependencies(self, experiment_id: str) -> Tuple[bool, List[str]]:
        """Check if all dependencies for an experiment are satisfied.
        
        Returns:
            Tuple of (all_satisfied, list_of_missing_dependencies)
        """
        deps = self.get_dependencies(experiment_id)
        missing = [
            dep for dep in deps
            if dep not in self.completed and dep not in self.failed
        ]
        return len(missing) == 0, missing
    
    def run_experiment(
        self,
        experiment_id: str,
        **kwargs,
    ) -> Optional[ExperimentResult]:
        """Run a single experiment.
        
        Args:
            experiment_id: The experiment ID to run (e.g., "S1", "R1")
            **kwargs: Additional arguments passed to the experiment runner
        
        Returns:
            ExperimentResult if successful, None if failed or skipped
        """
        # Check if already completed
        if experiment_id in self.completed:
            if self.verbose:
                print(f"Experiment {experiment_id} already completed")
            return self.completed[experiment_id]
        
        # Try to load existing result
        existing = self._load_existing_result(experiment_id)
        if existing is not None:
            self.completed[experiment_id] = existing
            return existing
        
        # Check dependencies
        deps_ok, missing = self.check_dependencies(experiment_id)
        if not deps_ok:
            if self.verbose:
                print(f"Cannot run {experiment_id}: missing dependencies {missing}")
            return None
        
        # Get the runner function
        if experiment_id not in _EXPERIMENT_RUNNERS:
            error = f"Unknown experiment ID: {experiment_id}"
            if self.verbose:
                print(error)
            self.failed[experiment_id] = error
            return None
        
        runner = _EXPERIMENT_RUNNERS[experiment_id]
        
        # Get experiment config
        config = self.registry.get(experiment_id)
        
        if self.verbose:
            print(f"Running experiment {experiment_id}: {config.name}")
        
        # Run the experiment
        start_time = datetime.utcnow()
        try:
            result = runner(**kwargs)
            end_time = datetime.utcnow()
            
            experiment_result = ExperimentResult(
                experiment_id=experiment_id,
                result=result,
                config={"experiment_id": experiment_id, **kwargs},
                start_time=start_time,
                end_time=end_time,
                status="COMPLETE",
            )
            
            # Save and track
            self._save_result(experiment_result)
            self.completed[experiment_id] = experiment_result
            
            if self.verbose:
                duration = (end_time - start_time).total_seconds()
                print(f"Completed {experiment_id} in {duration:.2f} seconds")
            
            return experiment_result
            
        except Exception as e:
            end_time = datetime.utcnow()
            error = str(e)
            
            experiment_result = ExperimentResult(
                experiment_id=experiment_id,
                result=None,
                config={"experiment_id": experiment_id, **kwargs},
                start_time=start_time,
                end_time=end_time,
                status="FAILED",
                error=error,
            )
            
            self._save_result(experiment_result)
            self.failed[experiment_id] = error
            
            if self.verbose:
                print(f"Failed {experiment_id}: {error}")
            
            return None
    
    def run_experiments(
        self,
        experiment_ids: List[str],
        **kwargs,
    ) -> Dict[str, Optional[ExperimentResult]]:
        """Run multiple experiments in dependency order.
        
        Args:
            experiment_ids: List of experiment IDs to run
            **kwargs: Additional arguments passed to each experiment runner
        
        Returns:
            Dict mapping experiment_id to result (or None if failed/skipped)
        """
        results = {}
        
        # Sort by dependency order (topological sort)
        sorted_ids = self._topological_sort(experiment_ids)
        
        for exp_id in sorted_ids:
            result = self.run_experiment(exp_id, **kwargs)
            results[exp_id] = result
        
        return results
    
    def _topological_sort(self, experiment_ids: List[str]) -> List[str]:
        """Sort experiment IDs in dependency order (dependencies first).
        
        Uses a simple greedy algorithm: repeatedly find experiments
        whose dependencies are all satisfied and add them to the order.
        """
        # Track which experiments are available (all dependencies satisfied)
        available = set()
        remaining = set(experiment_ids)
        order = []
        
        # Initially, experiments with no dependencies are available
        for exp_id in experiment_ids:
            deps = self.get_dependencies(exp_id)
            if len(deps) == 0:
                available.add(exp_id)
        
        # Build dependency graph
        dep_graph: Dict[str, List[str]] = {}
        for exp_id in experiment_ids:
            dep_graph[exp_id] = self.get_dependencies(exp_id)
        
        # Reverse graph: for each experiment, which experiments depend on it?
        reverse_graph: Dict[str, List[str]] = {exp_id: [] for exp_id in experiment_ids}
        for exp_id in experiment_ids:
            for dep in self.get_dependencies(exp_id):
                if dep in reverse_graph:
                    reverse_graph[dep].append(exp_id)
        
        # Greedy topological sort
        while available:
            # Pick an available experiment (sorted for determinism)
            exp_id = min(available)
            available.remove(exp_id)
            order.append(exp_id)
            
            # Make all experiments that depend on this one available
            for dependent in reverse_graph.get(exp_id, []):
                if dependent in remaining:
                    # Check if all dependencies are now satisfied
                    deps = dep_graph.get(dependent, [])
                    if all(dep in order for dep in deps):
                        available.add(dependent)
            
            remaining.discard(exp_id)
        
        # Add any remaining (should be empty if no cycles)
        if remaining:
            warnings.warn(f"Circular dependency detected, experiments {remaining} may be out of order")
            order.extend(sorted(remaining))
        
        return order
    
    def run_all_synthetic(self, **kwargs) -> Dict[str, Optional[ExperimentResult]]:
        """Run all synthetic experiments (S1-S6)."""
        synthetic_ids = self.registry.list_synthetic()
        return self.run_experiments(synthetic_ids, **kwargs)
    
    def run_all_real_data(self, **kwargs) -> Dict[str, Optional[ExperimentResult]]:
        """Run all real data experiments (R1-R14)."""
        real_ids = self.registry.list_real_data()
        return self.run_experiments(real_ids, **kwargs)
    
    def run_confirmatory(self, **kwargs) -> Dict[str, Optional[ExperimentResult]]:
        """Run all confirmatory experiments."""
        confirmatory_ids = self.registry.list_confirmatory()
        return self.run_experiments(confirmatory_ids, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {
            "completed": list(self.completed.keys()),
            "failed": self.failed.copy(),
            "total_experiments": len(self.registry.list_all()),
            "completed_count": len(self.completed),
            "failed_count": len(self.failed),
        }
    
    def get_results(self) -> Dict[str, ExperimentResult]:
        """Get all completed experiment results."""
        return self.completed.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all experiment results."""
        summary = {
            "total": len(self.completed),
            "synthetic": 0,
            "real_data": 0,
            "confirmatory": 0,
            "details": {},
        }
        
        synthetic_ids = self.registry.list_synthetic()
        real_ids = self.registry.list_real_data()
        confirmatory_ids = self.registry.list_confirmatory()
        
        for exp_id, result in self.completed.items():
            if exp_id in synthetic_ids:
                summary["synthetic"] += 1
            if exp_id in real_ids:
                summary["real_data"] += 1
            if exp_id in confirmatory_ids:
                summary["confirmatory"] += 1
            
            summary["details"][exp_id] = {
                "status": result.status,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "duration_seconds": (result.end_time - result.start_time).total_seconds(),
                "has_result": result.result is not None,
            }
        
        return summary


def run_experiment(
    experiment_id: str,
    **kwargs,
) -> Optional[ExperimentResult]:
    """Convenience function to run a single experiment.
    
    Creates a new executor and runs the specified experiment.
    
    Args:
        experiment_id: The experiment ID to run
        **kwargs: Additional arguments passed to the experiment runner
    
    Returns:
        ExperimentResult if successful, None otherwise
    """
    executor = ExperimentExecutor()
    return executor.run_experiment(experiment_id, **kwargs)


def run_all_synthetic(
    **kwargs,
) -> Dict[str, Optional[ExperimentResult]]:
    """Convenience function to run all synthetic experiments."""
    executor = ExperimentExecutor()
    return executor.run_all_synthetic(**kwargs)


def run_all_real_data(
    **kwargs,
) -> Dict[str, Optional[ExperimentResult]]:
    """Convenience function to run all real data experiments."""
    executor = ExperimentExecutor()
    return executor.run_all_real_data(**kwargs)
