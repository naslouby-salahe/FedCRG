# Next Action

**Last updated:** 2026-08-12
**Current time:** 2026-08-12

## Immediate Next Action

**Priority:** HIGH   
**Task:** Create and populate detailed matrix files per Section 3  
**Module:** docs/matrix/ and docs/FedCRG Audit and Implementation Matrix.md  
**Status:** IN PROGRESS (2/10 matrix files created)

### Specific Steps

Continue creating matrix files for:
1. Dataset specifications (03_dataset_requirements.md)
2. Training specifications (04_training_requirements.md)
3. Baseline suite (05_baseline_requirements.md)
4. Metrics and evaluation (06_metrics_requirements.md)
5. Experiments (07_experiment_requirements.md)
6. Implementation and artifacts (08_implementation_requirements.md)
7. Testing and validation (09_testing_requirements.md)
8. Failure states and claims (10_failure_claims.md)

### Specific Steps

Implement the CLI commands per Section 14.10.

### Why This is Next

CLI is needed to:
- Provide reproducible command-line execution per Section 14.10
- Enable `fedcrg verify` command for completeness checking
- Support automated execution of all experiments
- Provide user-friendly interface for running FedCRG

### Blocking Dependencies

None. All core modules and experiment registry are complete.

### After CLI

1. Create and populate detailed matrix files
2. Perform four matrix audits
3. Implement raw data symlink (data/raw)
4. Complete end-to-end integration testing

### Time Estimate

4-6 hours

### Resources Needed

- Roadmap Section 14.10 (Command-line execution contract)
- Existing all modules
