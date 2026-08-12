# Next Action

**Last updated:** 2026-08-12
**Current time:** 2026-08-12

## Immediate Next Action

**Priority:** HIGH   
**Task:** Create and populate detailed matrix files per Section 3  
**Module:** docs/matrix/ and docs/FedCRG Audit and Implementation Matrix.md  
**Status:** PARTIAL

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
