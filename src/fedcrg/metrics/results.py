"""Typed metric and policy-evaluation bundles."""
from dataclasses import dataclass
from fedcrg.core.enums import PolicyEvaluationStatus,PolicyId
@dataclass(frozen=True,slots=True)
class ClientMetrics:
    fpr:float; tpr:float; precision:float; recall:float; f1:float; auroc:float; auprc:float; band_error:float; high_excess:float; band_violation:float; absolute_fpr_error:float; attack_balanced_tpr:float|None=None
@dataclass(frozen=True,slots=True)
class PolicyEvaluation:
    client_id:str; policy:PolicyId; threshold:float|None; status:PolicyEvaluationStatus; metrics:ClientMetrics|None
