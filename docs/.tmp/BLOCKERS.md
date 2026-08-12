# Blockers

**Created:** 2026-08-12
**Status:** No active blockers

## Active Blockers

None currently identified.

## Resolved Blockers

None.

## Potential Future Blockers

### Data Access
- **Risk:** N-BaIoT and CIC IoT-DIAD 2024 datasets may require download/access
- **Mitigation:** Use existing data if available at `/home/naslouby/Projects/FedCRG/data/raw`
- **Status:** MONITORING
- **Owner:** Data infrastructure phase

### Computational Resources
- **Risk:** Federated training (30 rounds, 9 clients) may require significant compute
- **Mitigation:** Implement efficient training, use CPU if GPU unavailable
- **Status:** MONITORING
- **Owner:** Training implementation phase

### File Size Limits
- **Risk:** Large matrix files may exceed system limits
- **Mitigation:** Split matrix into multiple files (by section or category)
- **Status:** ACTIVE - Currently working around by splitting matrix
- **Owner:** Matrix extraction phase
- **Resolution:** Create matrix in parts (Part 1: Core, Part 2: Data, Part 3: Training, etc.)

## Blocked Tasks

None currently blocked.

## Dependencies

### External Dependencies (Not Blockers)
- Python 3.x with type hints
- NumPy (float64 support required)
- SciPy (special functions for Beta distribution)
- PyTorch (federated training)
- Pydantic (configuration validation)
- pandas (data manipulation)
- scikit-learn (metrics)

### Internal Dependencies (Sequencing)
1. Domain model must be complete before statistical core
2. Statistical core must be complete before data infrastructure
3. Data infrastructure must be complete before detector training
4. Detector training must be complete before policy evaluation
5. All components must be complete before experiments

## Escalation Path

1. Attempt to resolve using roadmap's internal precedence hierarchy
2. Record interpretation in AUDIT_LOG.md
3. Continue with narrowest interpretation preserving invariants
4. If truly blocking, mark in this file and continue other tasks
