"""Pre-data finite-sample protocol table generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.config.models import ExperimentConfig, ProtocolConfig
from fedcrg.core.types import OperatingBand
from fedcrg.protocol.mismatch import (
    clopper_pearson_interval,
    minimum_bidirectional_sample_count,
)
from fedcrg.protocol.readiness import ReadinessPlanCache, familywise_readiness_assurance


@dataclass(frozen=True, slots=True)
class MismatchCutoffCell:
    sample_count: int
    low_max_exceedances: int | None
    high_min_exceedances: int | None


class ProtocolTablePrecomputer:
    """Freeze all rank/cutoff tables needed by the registered real-data programme."""

    def precompute(self, config: ExperimentConfig, root: Path | None = None) -> tuple[Path, Path]:
        target_root = root or config.outputs_root / "cache" / "precomputed"
        target_root.mkdir(parents=True, exist_ok=True)
        readiness_path = target_root / "readiness_plans.json"
        mismatch_path = target_root / "mismatch_cutoffs.json"
        cache = ReadinessPlanCache(readiness_path)
        for sample_count, protocol in self._readiness_cells(config):
            cache.precompute(sample_count, protocol.band, protocol.readiness_assurance)

        mismatch_rows = [
            self._mismatch_cutoffs(sample_count, config.protocol.band, config.protocol.mismatch_confidence)
            for sample_count in (736, 1000, 1500, 2000, 3000)
        ]
        atomic_write_json(
            mismatch_path,
            {
                "band": {
                    "lower": config.protocol.band.lower,
                    "upper": config.protocol.band.upper,
                },
                "confidence": config.protocol.mismatch_confidence,
                "minimum_bidirectional_sample_count": minimum_bidirectional_sample_count(
                    config.protocol.band.lower,
                    config.protocol.mismatch_confidence,
                ),
                "cells": [
                    {
                        "sample_count": row.sample_count,
                        "low_max_exceedances": row.low_max_exceedances,
                        "high_min_exceedances": row.high_min_exceedances,
                    }
                    for row in mismatch_rows
                ],
            },
        )
        return readiness_path, mismatch_path

    @staticmethod
    def _readiness_cells(config: ExperimentConfig) -> tuple[tuple[int, ProtocolConfig], ...]:
        cells: dict[tuple[int, float, float, float], ProtocolConfig] = {}

        def add(sample_count: int, *, alpha: float, rho: float, assurance: float) -> None:
            payload = config.protocol.model_dump(mode="python")
            payload.update(
                {
                    "alpha": alpha,
                    "rho": rho,
                    "readiness_assurance": assurance,
                }
            )
            protocol = ProtocolConfig.model_validate(payload)
            key = (sample_count, alpha, rho, assurance)
            cells[key] = protocol

        for sample_count in (500, 1000, 1400, 1415, 1416, 1500, 2000, 3000):
            add(sample_count, alpha=0.01, rho=0.5, assurance=0.95)
        for rho in (0.25, 0.5, 1.0):
            add(2000, alpha=0.01, rho=rho, assurance=0.95)
        for alpha in (0.005, 0.01, 0.02, 0.05):
            add(2000, alpha=alpha, rho=0.5, assurance=0.95)
        for assurance in (0.90, 0.95, 0.99):
            add(2000, alpha=0.01, rho=0.5, assurance=assurance)
        add(2000, alpha=0.01, rho=0.5, assurance=familywise_readiness_assurance(9))
        add(1500, alpha=0.01, rho=0.5, assurance=0.95)
        return tuple((key[0], protocol) for key, protocol in sorted(cells.items()))

    @staticmethod
    def _mismatch_cutoffs(
        sample_count: int,
        band: OperatingBand,
        confidence: float,
    ) -> MismatchCutoffCell:
        lows: list[int] = []
        highs: list[int] = []
        for exceedances in range(sample_count + 1):
            interval = clopper_pearson_interval(exceedances, sample_count, confidence)
            if band.lower > 0.0 and interval.upper < band.lower:
                lows.append(exceedances)
            if interval.lower > band.upper:
                highs.append(exceedances)
        return MismatchCutoffCell(
            sample_count=sample_count,
            low_max_exceedances=max(lows) if lows else None,
            high_min_exceedances=min(highs) if highs else None,
        )
