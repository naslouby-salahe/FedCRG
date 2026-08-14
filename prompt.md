# FedCRG Light Cleanup — Specific Action List

## 1. Consolidate all filesystem layout code

Create one focused module:

`src/fedcrg/paths.py`

This becomes the **single place that knows how repository/runtime paths are constructed**.

Move path-only responsibilities there from:

* `config.py`
* `evidence/store.py`
* `data/preparation.py`
* `experiments/runner.py`
* `reporting.py`
* `cli.py`

Do **not** move file reading/writing, validation, hashing, serialization, or scientific logic into this module.

The rule should be:

> Business/scientific code asks a layout object for a path. It does not construct known repository paths itself.

---

## 2. Move `StudyPaths` out of `config.py`

`StudyPaths` is fundamentally a filesystem-layout value object, not scientific configuration logic.

Move it to `paths.py` and import it into `config.py`.

Keep its existing fields:

* `data_root`
* `preprocessed_root`
* `outputs_root`
* `results_root`

Do not change the YAML schema.

This keeps behavior unchanged while making path ownership clearer.

---

## 3. Add a configuration-file layout

Currently `Study.load()` directly contains:

* `Path("config/study.yaml")`
* `Path("config/datasets.yaml")`
* `Path("config/experiments.yaml")`

and `cli.campaign()` separately reconstructs `Path("config/study.yaml")`.

Introduce something like:

`ConfigLayout`

with properties for:

* config root
* study config
* datasets config
* experiments config

Then `Study.load()` and campaign work items both use the same source.

There should be exactly one definition of where these files live.

---

## 4. Make `PreparedLayout` a real root-aware layout

Currently `PreparedLayout` mostly contains filename strings such as:

* `manifest_filename`
* `preprocessing_filename`
* `eligibility_filename`
* `diad_eligibility_filename`
* `training_filename`
* `model_filename`
* `raw_staging_directory`
* `calibration_split_directory`
* `source_order_split_filename`

and callers repeatedly perform:

`root / PreparedLayout.manifest_filename`

This should become a proper layout object initialized with a root.

For example, conceptually:

`PreparedDatasetLayout(root)`

should expose:

* `manifest`
* `preprocessing`
* `eligibility`
* `diad_eligibility`
* `raw_staging`
* `splits`
* `seeded_splits`
* `source_order_split`

Callers should never need to combine these filenames manually.

---

## 5. Separate prepared-data paths from model-cache paths

`PreparedLayout` currently knows about both prepared dataset artifacts and model/training files.

That mixes two separate filesystem concepts.

Use separate small layout classes:

* `PreparedDatasetLayout`
* `ModelCacheLayout`
* `ScoreCacheLayout`
* `RunLayout`
* `ResultsBundleLayout`

This is still light cleanup because it only formalizes boundaries that already exist in the repository.

Do not introduce additional architecture layers beyond these concrete filesystem layouts.

---

## 6. Make `OutputsLayout` the entry point for output-related layouts

`OutputsLayout` is already close to being the authoritative output tree.

Keep it, but make it return specialized layouts instead of raw roots wherever appropriate.

For example:

* model cache for a model seed → `ModelCacheLayout`
* score cache for a model seed → `ScoreCacheLayout`
* run → `RunLayout`
* publication → publication layout
* campaign status → campaign path
* analysis result → analysis path

The caller should ask for the semantic artifact it needs instead of knowing the directory structure.

---

## 7. Stop manually constructing model-cache artifacts

Current training code does roughly:

`OutputsLayout(...).model_root(...)`

then manually appends:

* model filename
* training manifest filename

Instead, return a `ModelCacheLayout` containing:

* `root`
* `model`
* `training_manifest`

This removes repeated knowledge about model-cache contents.

---

## 8. Stop manually constructing score-cache artifacts

Use the same approach for scores.

`OutputsLayout.score_root(...)` should preferably return or feed a `ScoreCacheLayout` that knows:

* root
* descriptor
* manifest
* score artifacts

Any repeated score-cache filename joins should disappear from scoring code.

---

## 9. Move prepared-cache identity path construction into the layout layer

`PrepareData.prepared_root()` currently constructs:

`preprocessed_root / dataset / hash-prefix`

That is filesystem identity logic.

Move that construction into the path/layout module.

`PrepareData` should calculate the identities/hashes because that is preparation logic, but it should give those typed identities to the path object rather than decide directory syntax itself.

In other words:

* preparation decides **what identity**
* paths decide **where that identity lives**

---

## 10. Add a dataset-level prepared root accessor

The CLI currently manually builds:

`config.preprocessed_root / dataset.value`

when handling `preprocess --overwrite`.

That should come from the same prepared-data layout.

This prevents the CLI and `PrepareData` from having slightly different knowledge of the prepared tree.

---

## 11. Remove embedded nested paths from filename constants

Things like:

`"splits/seeded"`

and:

`"splits/source_order.json"`

should not be represented as filenames.

Expose structural properties instead:

* `splits`
* `seeded_splits`
* `source_order_split`

This avoids having path separators hidden inside string constants.

---

## 12. Consolidate publication paths

`PublicationPackageBuilder` currently gets the publication root and then manually creates:

* tables
* figures

even though layout enums already know these directories.

Introduce or reuse a small `PublicationLayout` exposing:

* root
* tables
* figures
* manifest

Reporting then consumes this layout directly.

---

## 13. Make `ResultsBundleLayout.required_directories` return paths

It currently exposes directory names and downstream verification reconstructs:

`destination / directory`

That reintroduces path composition.

Instead expose actual paths:

* metrics
* statistics
* tables
* figures
* reports
* provenance
* resolved configs

Verification should iterate those paths directly.

---

## 14. Centralize campaign result paths

`CampaignExecutor._build_results()` currently directly constructs:

`results_root / str(campaign_id)`

Results-building/reporting code knows the same convention separately.

Give the results root a campaign accessor so this rule exists once.

---

## 15. Centralize analysis-result paths

Both CLI execution and campaign execution construct:

`cache_analysis / f"{experiment_id.value}.json"`

Add an `analysis_result(experiment_id)` accessor.

This removes duplicated naming logic and ensures synthetic/benchmark output naming cannot drift.

---

## 16. Remove hardcoded provenance path strings

Reporting currently records:

`"data/preprocessed/"`

as a literal provenance value.

Use the actual configured/prepared root from `StudyPaths`.

A provenance record should never claim a path derived from a hardcoded string when the repository already has a typed path configuration.

---

## 17. Make path serialization consistent

There is currently a mixture of:

* `str(path)`
* `path.as_posix()`
* path objects

Choose one representation at serialization boundaries.

Given the existing output models, `PathString` with `as_posix()` is a good consistent choice.

Inside Python, retain `Path`.

Convert only when entering JSON/Pydantic output objects that require `PathString`.

---

## 18. Keep path creation separate from filesystem mutation

Do **not** turn `paths.py` into a filesystem service.

Layout classes should calculate paths.

Existing services should remain responsible for:

* `mkdir`
* `unlink`
* `rmtree`
* atomic writes
* validation
* checksums

That keeps the cleanup small and prevents a new god object.

---

# CLI cleanup

## 19. Clean the duplicated `RunLayout` import

`cli.py` imports from `fedcrg.evidence.store` once near the main imports and then imports `RunLayout` separately later.

Merge these into one import block.

---

## 20. Remove the local `RunManifest` import in `status()`

`status()` imports `RunManifest` from inside the loop.

Import it normally at module level.

There is no apparent circular dependency requiring the local import.

---

## 21. Use the typed `RunConfig` in `_purge_experiment_evidence`

The code currently:

* reads `run_config.json`
* uses `json.loads`
* receives a raw dictionary
* calls `.get("experiment_id")`
* catches every exception

A typed `RunConfig` model already exists.

Load the run config through `RunConfig.model_validate_json()` and compare its typed `experiment_id`.

That removes a primitive dictionary boundary and makes malformed evidence handling explicit.

---

## 22. Narrow broad exception handling in CLI status scanning

`status()` currently catches `Exception` while reading manifests.

Catch the expected validation/read failures instead.

For example, distinguish:

* malformed Pydantic artifact
* missing/unreadable file

Do not silently suppress arbitrary programming errors.

Keep the intended behavior of skipping corrupt/incomplete run artifacts.

---

## 23. Narrow broad exception handling in purge logic

Apply the same principle to `_purge_experiment_evidence()`.

A filesystem scan may legitimately encounter malformed/stale artifacts, but an unrelated programming exception should not silently cause the run to be ignored.

---

# Reporting cleanup

## 24. Remove duplicate manifest-reading implementations

Reporting currently has both `_run_manifest()` and separate code in `_completed_runs()` that parses `RunManifest` again.

Make `_completed_runs()` reuse `_run_manifest()`.

One helper should own tolerant manifest loading.

---

## 25. Reuse typed artifact loaders consistently

Where the repository already has helpers such as `load_json_model`, use them instead of repeating:

`Model.model_validate_json(path.read_text(...))`

Do this only where it genuinely reduces repetition.

Do not add wrapper methods around one-line operations just for abstraction.

---

## 26. Stop reloading `Study` inside reporting operations

Some reporting/result-bundle operations call `Study.load()` internally even though the study/configuration is already available higher in the execution flow.

Pass the existing `Study` or required typed configuration where reasonable.

Benefits:

* no hidden filesystem dependency
* tests become easier
* custom config paths remain respected
* one invocation uses one consistent loaded study

Do not rewrite every constructor just for dependency injection; target only hidden reloads in active workflows.

---

# Configuration cleanup

## 27. Remove the duplicate `DatasetCatalogue` docstring

`DatasetCatalogue` currently contains two consecutive class docstrings.

Keep one concise description.

---

## 28. Review the hardcoded policy-count check

The validator currently checks that the policy enum contains exactly `12` entries.

That duplicates protocol knowledge as a magic number.

Either:

* derive completeness from the declared policy registry, or
* represent the expected policy set explicitly as typed policy identities

Do not keep a naked `12` if the actual invariant is “the complete registered policy catalogue must be present.”

---

## 29. Keep scientific values untouched

Do not use this cleanup as an excuse to move or reinterpret:

* protocol alpha/rho
* statistical settings
* training settings
* seeds
* dataset eligibility rules
* thresholds
* experiment definitions
* policy semantics

The cleanup should produce identical resolved scientific configurations.

---

# Small value-object cleanup

## 30. Convert obvious boilerplate-only record classes to frozen dataclasses where appropriate

There are classes that only define an initializer and fields, such as small evidence/result records.

For these specific classes, consider:

`@dataclass(frozen=True, slots=True)`

instead of handwritten constructors.

Only do this when the class:

* has no Pydantic serialization requirement
* has no validation behavior
* has no mutable lifecycle

Do not mechanically convert all classes.

---

## 31. Keep Pydantic where validation or serialization matters

Do not replace Pydantic models such as:

* configuration models
* manifests
* CLI payloads
* persisted scientific evidence

The current usage is appropriate.

---

# Exception and corruption handling

## 32. Distinguish expected bad artifacts from programming failures

There are several `except Exception: continue` patterns in evidence/report scanning.

Review them individually.

Expected malformed historical/incomplete artifacts may be skipped.

Unexpected implementation failures should propagate.

The goal is not “never catch Exception”; it is “do not accidentally hide defects.”

---

## 33. Keep tolerant repository scanning where it is intentional

Status/report commands should still tolerate:

* incomplete run directories
* missing manifests
* malformed abandoned artifacts where explicitly intended

Do not make status inspection fragile just to make exceptions narrower.

---

# Naming and readability

## 34. Normalize `root`, `*_root`, and `layout` terminology

Use:

* `*_root` when the value is a `Path`
* `*_layout` when the value is a layout object

Avoid variables called `outputs` when they actually mean `outputs_root`.

For example:

`outputs = study.paths.outputs_root`

would be clearer as:

`outputs_root = study.paths.outputs_root`

This is a small but worthwhile repository-wide consistency pass.

---

## 35. Prefer semantic path variables

Use names such as:

* `manifest_path`
* `model_path`
* `run_root`
* `publication_root`

rather than generic:

* `path`
* `output`
* `destination`

when the semantic role is stable and obvious.

Do not rename temporary loop variables where `path` is already the clearest name.

---

## 36. Normalize `.exists()` versus `.is_file()` / `.is_dir()`

Where the code expects a file, use `is_file()`.

Where it expects a directory, use `is_dir()`.

Reserve `exists()` for cases where either type is acceptable.

This makes filesystem assumptions explicit.

---

# Import cleanup

## 37. Normalize imports after the path move

After creating `paths.py`:

* path/layout classes come from `fedcrg.paths`
* evidence persistence classes remain in `fedcrg.evidence.store`
* evidence models remain in `fedcrg.evidence.models`

This should make `evidence.store` substantially clearer because it will no longer contain both filesystem topology and persistence behavior.

---

## 38. Remove imports made obsolete by typed path APIs

Likely examples include some direct imports of:

* `Path`
* `LayoutDirectory`
* `LayoutArtifact`
* `json`

from modules that no longer construct or parse these things directly.

Do this after the migration rather than speculatively.

---

# Evidence/store cleanup

## 39. Make `evidence.store` actually about storage

After path extraction, `evidence/store.py` should primarily own:

* atomic writes
* manifest stores
* typed model loading
* evidence persistence
* artifact verification
* environment capture if still appropriate

It should not own the repository's entire directory taxonomy.

---

## 40. Remove manual path composition inside `ArtifactVerifier`

For example, verification currently constructs some paths using:

`layout.data / LayoutArtifact...`

If the artifact is part of `RunLayout`, expose it directly as a property.

`ArtifactVerifier` should map artifact types to semantic layout properties, not know artifact filenames.

---

# Tests

## 41. Add focused layout tests

Do not rewrite the test suite.

Add/adjust small tests proving that each layout resolves the expected paths.

Cover:

* config files
* prepared dataset artifacts
* model cache
* score cache
* run layout
* publication layout
* campaign results bundle
* analysis cache

These tests should make future ad-hoc path construction unnecessary.

---

## 42. Add a guard against known hardcoded repository paths

A lightweight architecture test can scan `src/fedcrg` for forbidden literals such as:

* `config/study.yaml`
* `config/datasets.yaml`
* `config/experiments.yaml`
* `data/preprocessed/`
* manually repeated output subdirectory names

Allow those strings only inside `paths.py` or configuration fixtures where appropriate.

This directly prevents the path logic from spreading again.

---

## 43. Test path behavior with temporary roots

Layout tests should use `tmp_path`.

Verify that custom:

* data root
* preprocessed root
* outputs root
* results root
* config root

propagate throughout the workflow without accidentally falling back to repository-relative literals.

This is especially important after removing hidden `Study.load()` calls.

---

# Scope safeguards

## 44. Preserve existing artifact names and directory structure

This cleanup should consolidate **construction**, not redesign the on-disk layout.

Do not rename existing:

* directories
* manifests
* cache keys
* run IDs
* evidence files
* publication files

unless an actual bug is discovered.

That avoids invalidating existing evidence.

---

## 45. Preserve CLI behavior

No command restructuring.

Keep the existing commands and options.

Only replace internal path construction with layout calls.

---

## 46. Preserve public scientific APIs

Do not rename domain classes or restructure packages as part of this work.

Path consolidation should be the only meaningful cross-module structural change.

---

## 47. No compatibility layer

Update callers directly.

Do not leave:

* deprecated path helpers
* aliases for moved layout classes
* forwarding imports
* old/new path implementations in parallel

Once migrated, there should be one path implementation.

---

# Final small cleanup pass

After path consolidation, perform one conservative repository pass for:

* duplicate imports
* duplicate docstrings
* dead imports
* obviously redundant helpers
* broad exceptions hiding programming failures
* repeated typed model-loading code
* obvious variable-name inconsistencies
* actual Ruff/Pyright findings
* obvious hardcoded path fragments
* path joins that should use layouts

Do **not** expand that pass into architecture redesign.

---

# Suggested resulting ownership

`src/fedcrg/paths.py`

owns:

* `StudyPaths`
* `ConfigLayout`
* directory/artifact enums if still useful
* `PreparedDatasetLayout`
* `ModelCacheLayout`
* `ScoreCacheLayout`
* `OutputsLayout`
* `RunLayout`
* `PublicationLayout`
* `ResultsBundleLayout`

`src/fedcrg/evidence/store.py`

owns:

* atomic persistence
* manifest stores
* typed artifact loading
* artifact verification

`src/fedcrg/config.py`

owns:

* scientific/runtime configuration models
* YAML loading
* validation
* experiment resolution

`src/fedcrg/data/preparation.py`

owns:

* preparation behavior
* cache identities/hashes
* dataset materialization

but **not path syntax**.

`src/fedcrg/reporting.py`

owns:

* report/table/figure generation
* result bundling

but **not output-tree construction**.

---

# Definition of done

The cleanup is done when:

1. A developer can look in one module and understand the complete filesystem layout.
2. Known repository paths are not manually reconstructed throughout the codebase.
3. `Study.load()` and campaign execution use the same configuration paths.
4. Prepared/model/score/run/results layouts each have clear ownership.
5. Existing filesystem structure remains unchanged.
6. Existing CLI behavior remains unchanged.
7. Existing scientific configuration and experiment semantics remain unchanged.
8. Existing artifacts remain readable.
9. No compatibility wrappers remain.
10. Ruff, Pyright/Pylance, and the relevant test suite pass.
11. The resulting code is smaller or clearer, not more abstract than before.
