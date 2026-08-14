from __future__ import annotations

import numpy as np

from fedcrg.types import Metric

_IQR_PERCENTILES = (25, 75)


def iqr(values: np.ndarray) -> Metric:
    lower, upper = np.percentile(values, _IQR_PERCENTILES)
    return float(upper - lower)
