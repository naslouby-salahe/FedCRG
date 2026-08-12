"""Typed experiment registry."""

from fedcrg.core.enums import ExperimentId
from fedcrg.experiments.definitions import definitions
from fedcrg.experiments.models import ExperimentDefinition


class ExperimentRegistry:
    def __init__(self) -> None:
        items = definitions()
        self._items = {item.id: item for item in items}
        if len(self._items) != len(items):
            raise RuntimeError("Duplicate experiment identifiers")
        for item in items:
            unknown = set(item.dependencies).difference(self._items)
            if unknown:
                raise RuntimeError(f"Unknown dependencies for {item.id}: {sorted(unknown)}")

    def get(self, experiment_id: ExperimentId) -> ExperimentDefinition:
        return self._items[experiment_id]

    def all(self) -> tuple[ExperimentDefinition, ...]:
        return tuple(self._items.values())
