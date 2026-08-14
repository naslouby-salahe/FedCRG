from __future__ import annotations

from fedcrg.evidence.store import PreparedDatasetManifestStore


def test_preprocessing_manifest_store_is_explicitly_available() -> None:
    assert PreparedDatasetManifestStore.__name__ == "PreparedDatasetManifestStore"
