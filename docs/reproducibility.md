# Reproducibility contract

Every run is identified by a deterministic ID derived from the resolved experiment configuration and explicit seeds. The resolved configuration is written before execution and hashed.

A completed run must include:

- resolved configuration and configuration hash;
- environment metadata;
- dataset/split manifests and integrity evidence;
- model/training manifest and model hash when training is used;
- score hashes;
- threshold decisions;
- metrics;
- artifact hash manifest;
- verification report.

Writes use temporary files followed by atomic replacement. A directory whose manifest status is `COMPLETE` must not be mutated by application code.
