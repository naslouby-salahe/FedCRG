"""Threshold-space shrinkage comparator and deterministic tuning rule."""

import numpy as np
from fedcrg.core.constants import SHRINKAGE_N0_CANDIDATES
from fedcrg.metrics.classification import confusion_matrix, fpr
from fedcrg.policies.base import ClientPolicyData, empirical_quantile

def tune_shrinkage(clients: tuple[ClientPolicyData, ...], alpha: float) -> int:
    best_n0 = SHRINKAGE_N0_CANDIDATES[0]; best_error = float("inf")
    for n0 in SHRINKAGE_N0_CANDIDATES:
        errors=[]
        for client in clients:
            local=empirical_quantile(client.calibration_scores,alpha); n=len(client.calibration_scores); weight=n/(n+n0); threshold=weight*local+(1-weight)*client.protocol.reference.value; labels=np.zeros(len(client.mismatch_scores),dtype=np.int64); estimate=fpr(confusion_matrix(client.mismatch_scores,labels,threshold)); errors.append(abs(estimate-alpha))
        mean_error=float(np.mean(errors))
        if mean_error<best_error or (np.isclose(mean_error,best_error) and n0>best_n0): best_error=mean_error; best_n0=n0
    return best_n0
def shrinkage(client:ClientPolicyData,alpha:float,n0:int)->float:
    local=empirical_quantile(client.calibration_scores,alpha); n=len(client.calibration_scores); weight=n/(n+n0); return weight*local+(1-weight)*client.protocol.reference.value
