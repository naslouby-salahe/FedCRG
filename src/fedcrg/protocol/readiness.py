"""Finite-sample local-readiness planning, lookup, and continuity diagnostics.

Observed score values never participate in rank optimization. The persistent table
is keyed only by the pre-data statistical contract `(n, a, b, assurance)`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import special

from fedcrg.core.enums import CalibrationReadinessState
from fedcrg.core.types import OperatingBand
from fedcrg.protocol.results import (
    CalibrationReadiness,
    ContinuityDiagnostics,
    ReadinessPlan,
)


def coverage_probability(
    rank: int,
    sample_count: int,
    band: OperatingBand,
) -> float:
    if not 1 <= rank <= sample_count:
        raise ValueError("rank must be inside [1, sample_count]")
    upper_shape = sample_count + 1 - rank
    lower_shape = rank
    probability = special.betainc(upper_shape, lower_shape, band.upper)
    probability -= special.betainc(upper_shape, lower_shape, band.lower)
    return float(probability)


class ReadinessPlanBuilder:
    """Choose the order-statistic rank using protocol constants only."""

    def build(
        self,
        sample_count: int,
        band: OperatingBand,
        assurance: float,
    ) -> ReadinessPlan:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if not 0.0 < assurance < 1.0:
            raise ValueError("assurance must be in (0,1)")
        best_rank = 1
        best_probability = -1.0
        for rank in range(1, sample_count + 1):
            probability = coverage_probability(rank, sample_count, band)
            if probability > best_probability or (
                math.isclose(
                    probability,
                    best_probability,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                and rank > best_rank
            ):
                best_rank = rank
                best_probability = probability
        state = (
            CalibrationReadinessState.READY
            if best_probability >= assurance
            else CalibrationReadinessState.NOT_READY
        )
        return ReadinessPlan(
            sample_count=sample_count,
            rank=best_rank,
            coverage_probability=best_probability,
            state=state,
            band=band,
            assurance=assurance,
        )


class ReadinessPlanCache:
    """Persistent pre-data plan table consumed by real-data policy evaluation.

    Persistence is intentionally tiny and dependency-free. Application code decides
    when the table is generated; runtime evaluation may only call :meth:`require`.
    """

    def __init__(
        self,
        path: Path | None = None,
        builder: ReadinessPlanBuilder | None = None,
    ) -> None:
        self.path = path
        self.builder = builder or ReadinessPlanBuilder()
        self._plans: dict[str, ReadinessPlan] = {}
        if path is not None and path.is_file():
            self._load(path)

    @staticmethod
    def key(
        sample_count: int,
        band: OperatingBand,
        assurance: float,
    ) -> str:
        return (
            f"n={sample_count}|a={band.lower:.17g}|b={band.upper:.17g}|"
            f"assurance={assurance:.17g}"
        )

    def precompute(
        self,
        sample_count: int,
        band: OperatingBand,
        assurance: float,
    ) -> ReadinessPlan:
        key = self.key(sample_count, band, assurance)
        candidate = self.builder.build(sample_count, band, assurance)
        existing = self._plans.get(key)
        if existing is not None and existing != candidate:
            raise RuntimeError("Frozen readiness-plan table is internally inconsistent")
        self._plans[key] = candidate
        if self.path is not None:
            self._save(self.path)
        return candidate

    def require(
        self,
        sample_count: int,
        band: OperatingBand,
        assurance: float,
    ) -> ReadinessPlan:
        key = self.key(sample_count, band, assurance)
        try:
            return self._plans[key]
        except KeyError as exc:
            raise FileNotFoundError(
                "Required pre-data readiness plan is absent. Run the protocol "
                "precomputation command before evaluating real scores: "
                + key
            ) from exc

    def _save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: {
                "sample_count": plan.sample_count,
                "rank": plan.rank,
                "coverage_probability": plan.coverage_probability,
                "state": plan.state.value,
                "band": {
                    "lower": plan.band.lower,
                    "upper": plan.band.upper,
                },
                "assurance": plan.assurance,
            }
            for key, plan in sorted(self._plans.items())
        }
        temp = path.with_name(f".{path.name}.tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def _load(self, path: Path) -> None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Readiness-plan table must be a JSON object")
        plans: dict[str, ReadinessPlan] = {}
        for key, item in raw.items():
            if not isinstance(item, dict):
                raise ValueError("Malformed readiness-plan entry")
            band_raw = item["band"]
            if not isinstance(band_raw, dict):
                raise ValueError("Malformed readiness-plan operating band")
            plan = ReadinessPlan(
                sample_count=int(item["sample_count"]),
                rank=int(item["rank"]),
                coverage_probability=float(item["coverage_probability"]),
                state=CalibrationReadinessState(str(item["state"])),
                band=OperatingBand(
                    float(band_raw["lower"]),
                    float(band_raw["upper"]),
                ),
                assurance=float(item["assurance"]),
            )
            expected_key = self.key(
                plan.sample_count,
                plan.band,
                plan.assurance,
            )
            if str(key) != expected_key:
                raise ValueError("Readiness-plan key does not match its payload")
            regenerated = self.builder.build(
                plan.sample_count,
                plan.band,
                plan.assurance,
            )
            if (
                regenerated.rank != plan.rank
                or regenerated.state is not plan.state
                or abs(
                    regenerated.coverage_probability
                    - plan.coverage_probability
                )
                > 1e-12
            ):
                raise ValueError("Readiness-plan table failed formula regeneration")
            plans[expected_key] = plan
        self._plans = plans


class CalibrationReadinessEvaluator:
    """Select the precomputed order statistic and audit continuity at that point."""

    def evaluate(
        self,
        scores: np.ndarray,
        plan: ReadinessPlan,
    ) -> CalibrationReadiness:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or len(values) != plan.sample_count:
            raise ValueError(
                "Observed calibration size does not match the frozen readiness plan"
            )
        if not np.isfinite(values).all():
            raise ValueError("Calibration scores must be finite")
        diagnostics = continuity_diagnostics(values, plan.rank)
        if plan.state is CalibrationReadinessState.NOT_READY:
            return CalibrationReadiness(
                plan=plan,
                threshold=None,
                tie_count=0,
                diagnostics=diagnostics,
            )
        ordered = np.sort(values, kind="stable")
        threshold = float(ordered[plan.rank - 1])
        tie_count = int(np.count_nonzero(ordered == threshold))
        return CalibrationReadiness(
            plan=plan,
            threshold=threshold,
            tie_count=tie_count,
            diagnostics=diagnostics,
        )


def continuity_diagnostics(
    scores: np.ndarray,
    selected_rank: int,
) -> ContinuityDiagnostics:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Continuity diagnostics require a non-empty score vector")
    ordered = np.sort(values, kind="stable")
    if not 1 <= selected_rank <= len(ordered):
        raise ValueError("selected_rank must lie inside the score vector")
    selected = float(ordered[selected_rank - 1])
    unique = np.unique(ordered)
    duplicate_count = int(len(ordered) - len(unique))
    positive_spacing = np.diff(unique)
    minimum_spacing = (
        None
        if len(positive_spacing) == 0
        else float(np.min(positive_spacing))
    )
    return ContinuityDiagnostics(
        unique_score_fraction=float(len(unique) / len(ordered)),
        duplicate_count=duplicate_count,
        selected_threshold_multiplicity=int(np.count_nonzero(ordered == selected)),
        minimum_positive_spacing=minimum_spacing,
    )


def familywise_readiness_assurance(
    client_count: int,
    familywise_alpha: float = 0.05,
) -> float:
    if client_count <= 0:
        raise ValueError("client_count must be positive")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must be in (0,1)")
    return 1.0 - familywise_alpha / client_count
