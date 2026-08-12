# Generated outputs

`outputs/` separates immutable scientific evidence from reusable caches.

- `runs/<run_id>/` is one immutable policy cell. After its manifest reaches `complete`, no file in the run may be mutated.
- `cache/datasets/` contains prepared deterministic dataset artifacts.
- `cache/models/` contains frozen detector checkpoints and training manifests keyed by data/config/model seed.
- `cache/scores/` contains hash-finalized Parquet score caches. Policy evaluation starts only after the cache hash is frozen.
- `cache/precomputed/` contains pre-data statistical lookup artifacts such as readiness-rank tables.
- `reports/latest/` and `reports/publication/` contain material regenerated exclusively from immutable score/threshold/metric artifacts.

A run contains `run_config.json`, `environment.json`, data/training/score evidence, threshold records, metric records, tables, figures, reports, logs, and verification hashes. Runtime data are ignored by Git; only this directory skeleton is versioned.
