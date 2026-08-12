# Next Action

**Last updated:** 2026-08-12
**Current time:** 2026-08-12

## Immediate Next Action

**Priority:** HIGH   
**Task:** Implement metrics module (fedcrg/metrics/)  
**File:** `fedcrg/metrics/` module  
**Status:** NOT STARTED

### Specific Steps

Implement the metrics module per Section 8.2 and hypotheses in Section 3:

1. **Core metrics from roadmap:**
   - Mean Excess Band Error (MEBE)
   - Band Violation Rate
   - Attack-Balanced Macro-TPR
   - AUROC and AUPRC (must be identical across threshold policies)
   - F1 scores for various thresholds

2. **Required metric properties:**
   - Deterministic computation
   - float64 precision
   - Proper handling of edge cases and NA values

3. **Key requirements from roadmap:**
   - H3: Any claimed FedCRG reliability gain must incur no more than 3.0 percentage-point absolute loss in Attack-Balanced Macro-TPR
   - H5: AUROC and AUPRC must be numerically identical across threshold policies using the same cached test scores, up to serialization/rounding tolerance of 1e-12
   - MEBE and BandViolationRate reduction compared to baselines

4. **Files to create:**
   - `fedcrg/metrics/__init__.py`
   - `fedcrg/metrics/base.py` - Base metric classes and interfaces
   - `fedcrg/metrics/classification.py` - FPR, TPR, F1, precision, recall
   - `fedcrg/metrics/band_metrics.py` - MEBE, BandViolationRate
   - `fedcrg/metrics/auc_metrics.py` - AUROC, AUPRC
   - `fedcrg/metrics/attack_balanced.py` - Attack-Balanced Macro-TPR
   - `fedcrg/metrics/verification.py` - Metric parity tests

### Why This is Next

Metrics are needed to:
- Evaluate FedCRG performance
- Compare against baselines
- Compute MEBE and BandViolationRate for hypothesis testing
- Verify H5 (AUROC/AUPRC invariance across threshold policies)

### Blocking Dependencies

None. All dependencies complete (scoring, preprocessing, baselines, FedCRG core).

### After Metrics

1. Implement experiment registry and execution pipeline
2. Implement CLI commands
3. Create and populate detailed matrix files
4. Perform four matrix audits
5. Implement raw data symlink (data/raw)
6. End-to-end integration testing

### Time Estimate

4-8 hours

### Resources Needed

- Roadmap Section 8.2 (Metrics)
- Roadmap Section 3 (Hypotheses)
- Existing scoring module for score access
- numpy and sklearn.metrics for implementation
