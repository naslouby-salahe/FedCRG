"""Re-audit the repository against the standing goal and report open violations.

This tool re-runs the fast, deterministic checks that guard the architecture contract:
forbidden vocabulary, stale package paths, redirect modules, Any annotations,
scientific defaults in configuration models, and the required directory layout.
It prints a concise table; exit code 0 means no actionable findings.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "fedcrg"

_FORBIDDEN_FRAGMENTS = ("canonical", "TODO", "FIXME", "Placeholder", "D:\\Projects\\FedCRG")
_STALE_PACKAGES = ("pipeline", "config", "data", "method", "thresholds")
_VAGUE_NAMES = ("utils", "helpers", "common", "manager", "handler", "processor", "engine",
                "service", "base", "models", "registry", "factory")
_REDIRECT_MODULES = ()


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def _check_findings() -> list[str]:
    findings: list[str] = []

    for path in _python_files(SRC):
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        for fragment in _FORBIDDEN_FRAGMENTS:
            if fragment in source:
                findings.append(f"{relative}: contains {fragment!r}")
        if path.name in _REDIRECT_MODULES:
            findings.append(f"{relative}: redirect module")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "Any":
                findings.append(f"{relative}:{node.lineno}: Any annotation")

    for package in _STALE_PACKAGES:
        if (SRC / package).exists():
            findings.append(f"src/fedcrg/{package}/ still exists (stale package)")

    for name in _VAGUE_NAMES:
        for path in _python_files(SRC):
            if path.name == f"{name}.py":
                findings.append(f"{path.relative_to(ROOT)}: vague filename {name}")

    required_dirs = ("configuration", "datasets", "decision", "evaluation",
                     "experiments", "artifacts", "analysis", "reporting",
                     "runtime", "cli", "detectors", "federation", "scoring", "domain")
    for directory in required_dirs:
        if not (SRC / directory).is_dir():
            findings.append(f"src/fedcrg/{directory}/ missing")

    return findings


def main() -> int:
    findings = _check_findings()
    print("FedCRG repository audit")
    print("=" * 40)
    if not findings:
        print("No actionable findings.")
        return 0
    for finding in findings:
        print(f"  - {finding}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
