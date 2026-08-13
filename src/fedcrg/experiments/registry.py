"""Lookup and validate the frozen experiment catalogue."""

from __future__ import annotations

from fedcrg.core.enums import ExperimentCode, ExperimentId
from fedcrg.experiments.definitions import definitions
from fedcrg.experiments.models import ExperimentDefinition


class ExperimentRegistry:
    def __init__(self) -> None:
        rows = definitions()
        self._definitions = {row.id: row for row in rows}
        if len(rows) != len(ExperimentCode) or len(self._definitions) != len(ExperimentCode):
            raise RuntimeError("The experiment registry must contain exactly S1-S6 and R1-R14")
        codes = {row.protocol_code for row in rows}
        if codes != set(ExperimentCode):
            raise RuntimeError("Experiment registry does not exactly cover S1-S6/R1-R14")

    def get(self, experiment_id: ExperimentId) -> ExperimentDefinition:
        return self._definitions[experiment_id]

    def all(self) -> tuple[ExperimentDefinition, ...]:
        return tuple(self._definitions.values())
