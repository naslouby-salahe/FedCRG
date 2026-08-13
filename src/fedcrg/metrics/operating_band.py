"""Metrics for operating-band reliability."""
from fedcrg.core.types import OperatingBand
def band_error(fpr_value:float,band:OperatingBand)->float:
    if fpr_value<band.lower:return band.lower-fpr_value
    if fpr_value>band.upper:return fpr_value-band.upper
    return 0.0
def high_excess(fpr_value:float,band:OperatingBand)->float:return max(0.0,fpr_value-band.upper)
def band_violation(fpr_value:float,band:OperatingBand)->float:return float(not band.contains(fpr_value))
def absolute_fpr_error(fpr_value:float,alpha:float)->float:return abs(fpr_value-alpha)
