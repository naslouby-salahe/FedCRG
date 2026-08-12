"""
FedCRG Metrics Module

Implements all metrics per Section 10 of the FedCRG Roadmap v2.0.

Normative reference: Section 10 (Metric registry)
"""

from fedcrg.metrics.band_metrics import (
    BandError,
    MEBEResult,
    HighExcessResult,
    BandViolationRateResult,
    MAFEResult,
    compute_band_error,
    compute_mebe,
    compute_high_excess,
    compute_band_violation_rate,
    compute_mafe,
)
from fedcrg.metrics.classification import (
    FPR,
    TPR,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    compute_fpr,
    compute_tpr,
    compute_precision,
    compute_recall,
    compute_f1,
)
from fedcrg.metrics.auc_metrics import (
    AUROC,
    AUPRC,
    AUROCResult,
    AUPRCResult,
    compute_auroc,
    compute_auprc,
    verify_auc_invariance,
)
from fedcrg.metrics.attack_balanced import (
    ABTPR,
    ABMacroTPR,
    AttackBalancedTPR,
    AttackGroupTPR,
    ClientABTPR,
    compute_abmacro_tpr,
    compute_client_abmacro_tpr,
    compute_attack_group_tpr,
)

__all__ = [
    # Band metrics
    "BandError",
    "MEBEResult",
    "HighExcessResult",
    "BandViolationRateResult",
    "MAFEResult",
    "compute_band_error",
    "compute_mebe",
    "compute_high_excess",
    "compute_band_violation_rate",
    "compute_mafe",
    # Classification metrics
    "FPR",
    "TPR",
    "Precision",
    "Recall",
    "F1Score",
    "ConfusionMatrix",
    "compute_fpr",
    "compute_tpr",
    "compute_precision",
    "compute_recall",
    "compute_f1",
    "compute_fpr_from_counts",
    "compute_tpr_from_counts",
    # AUC metrics
    "AUROC",
    "AUPRC",
    "AUROCResult",
    "AUPRCResult",
    "compute_auroc",
    "compute_auprc",
    "verify_auc_invariance",
    # Attack-balanced metrics
    "ABTPR",
    "ABMacroTPR",
    "AttackBalancedTPR",
    "AttackGroupTPR",
    "ClientABTPR",
    "compute_abmacro_tpr",
    "compute_client_abmacro_tpr",
    "compute_attack_group_tpr",
]
