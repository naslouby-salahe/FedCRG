"""Immutable run artifact verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fedcrg.artifacts.hashing import sha256_file
from fedcrg.artifacts.layout import RunLayout
from fedcrg.artifacts.serialization import atomic_write_json


@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool
    missing: tuple[str, ...]
    mismatched: tuple[str, ...]
    hashes: dict[str, str]


class ArtifactVerifier:
    """Records evidence before completion and verifies it read-only afterwards."""

    def _required_files(self, layout: RunLayout) -> tuple[Path, ...]:
        return (layout.resolved_config, layout.environment)

    def _hashable_files(self, layout: RunLayout) -> tuple[Path, ...]:
        # The lifecycle manifest changes from VERIFYING to COMPLETE after evidence is recorded.
        return tuple(
            sorted(
                (
                    path
                    for path in layout.root.rglob("*")
                    if path.is_file()
                    and path != layout.manifest
                    and layout.verification not in path.parents
                ),
                key=lambda path: str(path.relative_to(layout.root)),
            )
        )

    def record(self, layout: RunLayout) -> VerificationResult:
        missing = tuple(
            str(path.relative_to(layout.root))
            for path in self._required_files(layout)
            if not path.exists()
        )
        hashes = {
            str(path.relative_to(layout.root)): sha256_file(path)
            for path in self._hashable_files(layout)
        }
        result = VerificationResult(not missing, missing, (), hashes)
        layout.verification.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            layout.verification / "hashes.json",
            {"missing": list(missing), "hashes": hashes},
        )
        return result

    def verify(self, layout: RunLayout) -> VerificationResult:
        evidence_path = layout.verification / "hashes.json"
        if not evidence_path.exists():
            return VerificationResult(False, ("verification/hashes.json",), (), {})
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        expected: dict[str, str] = evidence.get("hashes", {})
        missing = tuple(sorted(path for path in expected if not (layout.root / path).exists()))
        mismatched = tuple(
            sorted(
                path
                for path, expected_hash in expected.items()
                if (layout.root / path).exists()
                and sha256_file(layout.root / path) != expected_hash
            )
        )
        return VerificationResult(not missing and not mismatched, missing, mismatched, expected)
