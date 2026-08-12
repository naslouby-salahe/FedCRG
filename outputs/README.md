# Output Layout

`outputs/` separates immutable scientific runs from reusable caches and assembled publication reports.

```text
outputs/
├── runs/<run_id>/
│   ├── manifest.json
│   ├── resolved_config.yaml
│   ├── environment.json
│   ├── data/
│   ├── training/
│   ├── scores/
│   ├── decisions/
│   ├── metrics/
│   ├── tables/
│   ├── figures/
│   ├── reports/
│   ├── logs/
│   └── verification/
├── cache/
│   ├── datasets/
│   ├── models/
│   ├── scores/
│   └── precomputed/
└── reports/
    ├── latest/
    └── publication/
```

A run becomes immutable when its lifecycle reaches `complete`. Verification of a completed run is read-only. Cache entries may be rebuilt and are never treated as publication evidence.
