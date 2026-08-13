# Generated outputs

`outputs/` separates immutable scientific evidence from reusable caches.

- `logs/` contains runtime logs.
- `monitoring/` contains runtime resource-monitoring samples (CPU/RAM/VRAM).
- `runs/<run_id>/` is one immutable policy cell. After its manifest reaches `complete`, no file in the run may be mutated.
- `cache/models/` contains frozen detector checkpoints and training manifests keyed by data/config/model seed.
- `cache/scores/` contains hash-finalized Parquet score caches. Policy evaluation starts only after the cache hash is frozen.
- `cache/analysis/` contains pre-data statistical lookup artifacts such as readiness-rank tables.
- `reports/latest/` and `reports/publication/` contain material regenerated exclusively from immutable score/threshold/metric artifacts.

Preprocessed datasets live under `data/preprocessed/<dataset_id>/<identity>/`, not under
`outputs/`. Compatible experiments reuse the same preprocessed root by identity.

A run contains `run_config.json`, `environment.json`, data/training/score evidence, threshold records, metric records, tables, figures, reports, logs, and verification hashes. Runtime data are ignored by Git; only this directory skeleton is versioned.
