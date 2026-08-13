"""Finite-sample calibration-readiness planning and continuity diagnostics.

Rank optimization is deliberately separated from observed score evaluation. A
ReadinessPlanCache is populated from sample counts and protocol parameters only;
client scores are never passed to the optimizer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import special

from fedcrg.artifacts.serialization import atomic_write_json
from fedcrg.core.enums import CalibrationReadinessState
from fedcrg.core.types import OperatingBand
from fedcrg.protocol.results import (
    CalibrationReadiness,
    ContinuityDiagnostics,
    ReadinessPlan,
)


def coverage_probability(rank: int, sample_count: int, band: OperatingBand) -> float:
    if not 1 <= rank <= sample_count:
        raise ValueError("rank must be inside [1, sample_count]")
    upper_shape = sample_count + 1 - rank
    lower_shape = rank
    probability = special.betainc(upper_shape, lower_shape, band.upper)
    probability -= special.betainc(upper_shape, lower_shape, band.lower)
    return float(probability)


def induced_fpr_mean(rank: int, sample_count: int) -> float:
    return (sample_count + 1 - rank) / (sample_count + 1)


def induced_fpr_variance(rank: int, sample_count: int) -> float:
    numerator = (sample_count + 1 - rank) * rank
    denominator = (sample_count + 1) ** 2 * (sample_count + 2)
    return numerator / denominator


class ReadinessPlanBuilder:
    def build(self, sample_count: int, band: OperatingBand, assurance: float) -> ReadinessPlan:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if not 0.0 < assurance < 1.0:
            raise ValueError("assurance must be in (0, 1)")

        best_rank = 1
        best_probability = -1.0
        for rank in range(1, sample_count + 1):
            probability = coverage_probability(rank, sample_count, band)
            is_tie = math.isclose(
                probability,
                best_probability,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            if probability > best_probability or (is_tie and rank > best_rank):
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
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._plans: dict[str, ReadinessPlan] = {}
        if path is not None and path.exists():
            self._load(path)

    @staticmethod
    def _key(sample_count: int, band: OperatingBand, assurance: float) -> str:
        return (
            f"n={sample_count}|a={band.lower:.17g}|b={band.upper:.17g}|"
            f"assurance={assurance:.17g}"
        )

    def get(self, sample_count: int, band: OperatingBand, assurance: float) -> ReadinessPlan | None:
        return self._plans.get(self._key(sample_count, band, assurance))

    def precompute(
        self,
        sample_count: int,
        band: OperatingBand,
        assurance: float,
        builder: ReadinessPlanBuilder | None = None,
    ) -> ReadinessPlan:
        key = self._key(sample_count, band, assurance)
        existing = self._plans.get(key)
        if existing is not None:
            return existing
        plan = (builder or ReadinessPlanBuilder()).build(sample_count, band, assurance)
        self._plans[key] = plan
        if self.path is not None:
            self.save()
        return plan

    def require(self, sample_count: int, band: OperatingBand, assurance: float) -> ReadinessPlan:
        plan = self.get(sample_count, band, assurance)
        if plan is None:
            raise KeyError(
                "Readiness plan was not precomputed for the requested protocol cell"
            )
        return plan

    def save(self) -> None:
        if self.path is None:
            raise ValueError("Cannot persist a readiness cache without a path")
        rows = []
        for key in sorted(self._plans):
            plan = self._plans[key]
            rows.append(
                {
                    "key": key,
                    "n": plan.sample_count,
                    "rank_r": plan.rank,
                    "coverage_probability": plan.coverage_probability,
                    "ready": plan.state is CalibrationReadinessState.READY,
                    "a": plan.band.lower,
                    "b": plan.band.upper,
                    "assurance": plan.assurance,
                    "induced_fpr_mean": induced_fpr_mean(plan.rank, plan.sample_count),
                    "induced_fpr_variance": induced_fpr_variance(plan.rank, plan.sample_count),
                }
            )
        atomic_write_json(self.path, {"entries": rows})

    def _load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("entries", []):
            plan = ReadinessPlan(
                sample_count=int(row["n"]),
                rank=int(row["rank_r"]),
                coverage_probability=float(row["coverage_probability"]),
                state=(
                    CalibrationReadinessState.READY
                    if bool(row["ready"])
                    else CalibrationReadinessState.NOT_READY
                ),
                band=OperatingBand(float(row["a"]), float(row["b"])),
                assurance=float(row["assurance"]),
            )
            self._plans[str(row["key"])] = plan


class CalibrationReadinessEvaluator:
    def evaluate(self, scores: np.ndarray, plan: ReadinessPlan) -> CalibrationReadiness:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("Calibration scores must be one-dimensional")
        if len(values) != plan.sample_count:
            raise ValueError("Observed calibration size does not match the precomputed plan")
        if not np.isfinite(values).all():
            raise ValueError("Calibration scores must all be finite")

        ordered = np.sort(values, kind="stable")
        unique = np.unique(ordered)
        duplicate_count = len(ordered) - len(unique)
        positive_differences = np.diff(unique)
        positive_differences = positive_differences[positive_differences > 0]
        minimum_spacing = (
            float(np.min(positive_differences)) if len(positive_differences) else None
        )

        if plan.state is CalibrationReadinessState.NOT_READY:
            diagnostics = ContinuityDiagnostics(
                unique_score_fraction=float(len(unique) / len(ordered)),
                duplicate_count=duplicate_count,
                selected_threshold_multiplicity=0,
                minimum_positive_spacing=minimum_spacing,
            )
            return CalibrationReadiness(plan=plan, threshold=None, diagnostics=diagnostics)

        threshold = float(ordered[plan.rank - 1])
        multiplicity = int(np.count_nonzero(ordered == threshold))
        diagnostics = ContinuityDiagnostics(
            unique_score_fraction=float(len(unique) / len(ordered)),
            duplicate_count=duplicate_count,
            selected_threshold_multiplicity=multiplicity,
            minimum_positive_spacing=minimum_spacing,
        )
        return CalibrationReadiness(
            plan=plan,
            threshold=threshold,
            diagnostics=diagnostics,
        )


def familywise_readiness_assurance(client_count: int, family_error: float = 0.05) -> float:
    if client_count <= 0:
        raise ValueError("client_count must be positive")
    if not 0.0 < family_error < 1.0:
        raise ValueError("family_error must be in (0,1)")
    return 1.0 - family_error / client_count
