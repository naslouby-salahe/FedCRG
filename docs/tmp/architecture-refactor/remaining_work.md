# Remaining work (resumable checklist)

Update this file as phases complete. Status values: TODO / IN_PROGRESS / DONE.

- [ ] Phase 1: domain + config          STATUS: TODO
- [ ] Phase 2: data + detectors          STATUS: TODO
- [ ] Phase 3: federation + scoring      STATUS: TODO
- [ ] Phase 4: method + thresholds       STATUS: TODO
- [ ] Phase 5: evaluation + analysis/reporting split   STATUS: TODO
- [ ] Phase 6: application/ removed -> pipeline/ + experiments/definitions/   STATUS: TODO
- [ ] Phase 7: artifacts consolidation   STATUS: TODO
- [ ] Phase 8: reporting + cli           STATUS: TODO
- [ ] Phase 9: final hostile audit + full validation suite   STATUS: TODO

Each phase, when started, must:
1. Move/merge/rewrite the modules per migration_map.md.
2. Update every caller (grep repo-wide for old dotted path).
3. Adapt/rewrite the relevant tests (see current_state.md section 2 for old test->module map).
4. Delete the old files/packages -- no re-export shims.
5. Run `ruff check`, `ruff format --check`, `mypy`, targeted `pytest` subset for touched area.
6. Update this file's status and commit.

Do not run full pytest/mypy/ruff after every micro-edit -- only at phase completion.

## Notes / open decisions to resolve during implementation
- config/validate.py must not import thresholds/ (downward dependency violation in old
  config/validation.py -> policies.registry). Resolve by validating only against domain
  enums/config shape in config/, and doing threshold-set-completeness checks (if needed)
  in pipeline/ or thresholds/selection.py instead.
- application/* -> analysis/* imports (benchmark, claims, synthetic, report) currently run
  "backwards" vs target chain (pipeline should sit above analysis). Resolve during Phase 5/6:
  analysis kernels that pipeline/experiments actually need at execution time (e.g.
  analysis.robustness stress generators, analysis.benchmark harness) must be relocated to
  experiments/definitions/ or pipeline/, not left in analysis/ and imported upward.
- thresholds/comparators/reference_quantile.py existence TBD -- confirm whether REF-Q99-R
  policy is a thin wrapper over method/reference_threshold.py or a genuinely distinct
  comparator; only create the file if there is real distinct behavior.
- Local (function-body) imports scattered in cli/*.py and application/*.py exist to dodge
  circular imports -- fix root dependency direction during migration; do not preserve the
  workaround pattern in new code.
- scoring/__init__.py and data/datasets/__init__.py currently re-export symbols; make all
  __init__.py files empty (or minimal package docstring only) after migration.
- pyproject.toml: no pyrightconfig.json exists; prompt.md requires Pyright/Pylance-compatible
  type checking. Existing mypy config is strict (disallow_untyped_defs etc.) -- treat mypy
  as the working proxy for Pyright compatibility during phase validation; add pyright as a
  final-audit check if available in the environment.
- pytest-xdist is installed but not wired into addopts; use `pytest -n auto` explicitly for
  parallel runs during validation, per prompt.md's "pytest in parallel where supported".
