"""Small shared statistical helpers."""

from __future__ import annotations

import numpy as np

from fedcrg.types import Metric, Percentage


def iqr(values: np.ndarray, percentiles: tuple[Percentage, Percentage]) -> Metric:
    """Interquartile (or other requested percentile) range of `values`."""
    lower, upper = np.percentile(values, percentiles)
    return float(upper - lower)
