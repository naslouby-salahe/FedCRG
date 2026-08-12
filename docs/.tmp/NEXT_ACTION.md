# Next Action

**Last updated:** 2026-08-12
**Current time:** 2026-08-12

## Immediate Next Action

**Priority:** HIGH  
**Task:** Complete remaining implementation and verification per Section 8 of prompt.md  
**Status:** All four audits (1-4) completed successfully ✅

### Next Phase: Execution and Verification

Now that all four audits have passed, the next priority is:

1. **Complete synthetic experiments** (S2-S5) - High priority
   - S1 and S6 already completed
   - Need to run S2 (target-FPR sensitivity), S3 (temporal dependence), S4 (calibration shift), S5 (contamination)
   
2. **Implement real data experiments** (R1-R14) - Medium priority
   - All functions exist, need to ensure they can run with actual data
   
3. **Complete testing suite** - Medium priority
   - Add leakage tests, invariance tests, metrics tests
   - Ensure all tests pass
   
4. **Final verification** - High priority
   - Run `fedcrg verify` command
   - Ensure all required artifacts are present
   - Verify claim gates (G0-G8)

### Why This is Next

Per prompt.md Section 8 (Implementation strategy) and Section 15 (Experiment execution order):
- Audits 1-4 are complete
- Core implementation is verified
- Now need to execute experiments and complete verification
- This establishes that the implementation is not just structurally complete but also functionally correct

### Blocking Dependencies

None. All audits passed, all modules implemented, all experiments registered.

### After Execution

1. Run `fedcrg verify` to check completeness
2. Complete final hostile audits (Section 21)
3. Generate required tables and figures (Section 17)
4. Build final reports

### Time Estimate

6-8 hours for remaining implementation and verification

### Resources Needed

- Roadmap Sections 15-17 (Execution order, tables/figures)
- All existing modules
- Raw data via data/raw symlink

### Verification Checklist

- [ ] All synthetic experiments (S1-S6) executed and results saved
- [ ] All real data experiments (R1-R14) can execute
- [ ] `fedcrg verify` passes
- [ ] All required artifacts present
- [ ] All claim gates (G0-G8) evaluated
- [ ] All required tables/figures generated
- [ ] Final hostile audit passes
