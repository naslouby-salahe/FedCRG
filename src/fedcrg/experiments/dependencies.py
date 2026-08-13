"""Dependency evaluation and deterministic topological ordering."""

from __future__ import annotations

from fedcrg.domain.enums import ExperimentId, ExperimentStatus
from fedcrg.experiments.registry import ExperimentRegistry


class DependencyResolver:
    def __init__(self, registry: ExperimentRegistry) -> None:
        self.registry = registry

    def blockers(
        self,
        experiment_id: ExperimentId,
        statuses: dict[ExperimentId, ExperimentStatus],
    ) -> tuple[ExperimentId, ...]:
        blockers = []
        for dependency in self.registry.get(experiment_id).dependencies:
            if statuses.get(dependency) is not ExperimentStatus.COMPLETE:
                blockers.append(dependency)
        return tuple(blockers)

    def order(self, experiment_ids: tuple[ExperimentId, ...]) -> tuple[ExperimentId, ...]:
        requested = set(experiment_ids)
        expanded = set(requested)
        stack = list(requested)
        while stack:
            current = stack.pop()
            for dependency in self.registry.get(current).dependencies:
                if dependency not in expanded:
                    expanded.add(dependency)
                    stack.append(dependency)
        ordered: list[ExperimentId] = []
        remaining = set(expanded)
        while remaining:
            ready = sorted(
                (
                    item
                    for item in remaining
                    if all(dep in ordered for dep in self.registry.get(item).dependencies)
                ),
                key=str,
            )
            if not ready:
                raise ValueError("Experiment dependency graph contains a cycle")
            for item in ready:
                ordered.append(item)
                remaining.remove(item)
        return tuple(item for item in ordered if item in expanded)
