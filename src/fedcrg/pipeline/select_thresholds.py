"""Pre-data finite-sample protocol table generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.json_io import atomic_write_json
from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.config.method_config import ProtocolConfig
from fedcrg.domain.enums import ExperimentAxisId, ExperimentId
from fedcrg.domain.values import OperatingBand
from fedcrg.experiments.experiment_definition import get_experiment_definition
from fedcrg.method.mismatch_detection import (
    clopper_pearson_interval,
    minimum_bidirectional_sample_count,
)
from fedcrg.method.calibration_readiness import (
    ReadinessPlanCache,
    familywise_readiness_assurance,
)


@dataclass(frozen=True, slots=True)
class MismatchCutoffCell:
    sample_count: int
    low_max_exceedances: int | None
    high_min_exceedances: int | None


class ProtocolTablePrecomputer:
    """Freeze every readiness/cutoff entry needed by the registered programme."""

    def precompute(
        self,
        config: ExperimentConfig,
        root: Path | None = None,
    ) -> tuple[Path, Path]:
        target_root = root or config.outputs_root / "cache" / "precomputed"
        target_root.mkdir(parents=True, exist_ok=True)
        readiness_path = target_root / "readiness_plans.json"
        mismatch_path = target_root / "mismatch_cutoffs.json"
        cache = ReadinessPlanCache(readiness_path)
        for sample_count, protocol in self._readiness_cells(config):
            cache.precompute(
                sample_count,
                protocol.band,
                protocol.readiness_assurance,
            )

        mismatch_definition = get_experiment_definition(ExperimentId.MISMATCH_POWER)
        mismatch_counts = tuple(
            int(value) for value in mismatch_definition.axis(ExperimentAxisId.MISMATCH_N).values
        )
        mismatch_rows = tuple(
            self._mismatch_cutoffs(
                sample_count,
                config.protocol.band,
                config.protocol.mismatch_confidence,
            )
            for sample_count in mismatch_counts
        )
        atomic_write_json(
            mismatch_path,
            {
                "band": config.protocol.band,
                "confidence": config.protocol.mismatch_confidence,
                "minimum_bidirectional_sample_count": minimum_bidirectional_sample_count(
                    config.protocol.band.lower,
                    config.protocol.mismatch_confidence,
                ),
                "cells": mismatch_rows,
            },
        )
        return readiness_path, mismatch_path

    def _readiness_cells(
        self,
        config: ExperimentConfig,
    ) -> tuple[tuple[int, ProtocolConfig], ...]:
        cells: dict[tuple[int, float, float, float], ProtocolConfig] = {}

        def add(
            sample_count: int,
            *,
            alpha: float,
            rho: float,
            assurance: float,
        ) -> None:
            protocol = ProtocolConfig(
                id=config.protocol.id,
                version=config.protocol.version,
                alpha=alpha,
                rho=rho,
                readiness_assurance=assurance,
                mismatch_confidence=config.protocol.mismatch_confidence,
                strict_exceedance=config.protocol.strict_exceedance,
                reject_calibration_ties=config.protocol.reject_calibration_ties,
            )
            cells[(sample_count, alpha, rho, assurance)] = protocol

        primary_n = config.dataset.split.calibration_benign
        add(
            primary_n,
            alpha=config.protocol.alpha,
            rho=config.protocol.rho,
            assurance=config.protocol.readiness_assurance,
        )

        for value in (
            get_experiment_definition(ExperimentId.READINESS_SAMPLE_SIZE)
            .axis(ExperimentAxisId.CALIBRATION_N)
            .values
        ):
            add(
                int(value),
                alpha=config.protocol.alpha,
                rho=config.protocol.rho,
                assurance=config.protocol.readiness_assurance,
            )

        for value in (
            get_experiment_definition(ExperimentId.TOLERANCE_SENSITIVITY)
            .axis(ExperimentAxisId.RHO)
            .values
        ):
            add(
                primary_n,
                alpha=config.protocol.alpha,
                rho=float(value),
                assurance=config.protocol.readiness_assurance,
            )

        for value in (
            get_experiment_definition(ExperimentId.TARGET_FPR_REAL)
            .axis(ExperimentAxisId.ALPHA)
            .values
        ):
            add(
                primary_n,
                alpha=float(value),
                rho=config.protocol.rho,
                assurance=config.protocol.readiness_assurance,
            )

        for value in (
            get_experiment_definition(ExperimentId.ASSURANCE_SENSITIVITY)
            .axis(ExperimentAxisId.READINESS_ASSURANCE)
            .values
        ):
            add(
                primary_n,
                alpha=config.protocol.alpha,
                rho=config.protocol.rho,
                assurance=float(value),
            )

        add(
            primary_n,
            alpha=config.protocol.alpha,
            rho=config.protocol.rho,
            assurance=familywise_readiness_assurance(
                config.dataset.expected_clients or config.dataset.minimum_clients
            ),
        )

        if config.dataset.id.value == "nbaiot":
            add(
                1500,
                alpha=config.protocol.alpha,
                rho=config.protocol.rho,
                assurance=config.protocol.readiness_assurance,
            )

        for cell in get_experiment_definition(ExperimentId.TARGET_FPR_SYNTHETIC).coupled_cells:
            add(
                int(cell.value(ExperimentAxisId.CALIBRATION_N)),
                alpha=float(cell.value(ExperimentAxisId.ALPHA)),
                rho=config.protocol.rho,
                assurance=config.protocol.readiness_assurance,
            )
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
            interval = clopper_pearson_interval(
                exceedances,
                sample_count,
                confidence,
            )
            if band.lower > 0.0 and interval.upper < band.lower:
                lows.append(exceedances)
            if interval.lower > band.upper:
                highs.append(exceedances)
        return MismatchCutoffCell(
            sample_count=sample_count,
            low_max_exceedances=max(lows) if lows else None,
            high_min_exceedances=min(highs) if highs else None,
        )
