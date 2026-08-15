"""SHA-256 hashing helpers used for content-addressed caching."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fedcrg.types import ByteCount, Sha256


def sha256_file(path: Path, chunk_size: ByteCount = 1024 * 1024) -> Sha256:
    """Hash a file's bytes in fixed-size chunks so caching works on files too large to load whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> Sha256:
    """Hash a UTF-8 encoded string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
