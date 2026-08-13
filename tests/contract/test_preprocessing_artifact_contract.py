from __future__ import annotations

from fedcrg.artifacts.manifests import PreprocessingManifestStore


def test_preprocessing_manifest_store_is_explicitly_available() -> None:
    assert PreprocessingManifestStore.__name__ == "PreprocessingManifestStore"
