"""Re-audit the repository against the standing goal and report open violations.

This tool re-runs the fast, deterministic checks that guard the architecture
contract: forbidden vocabulary, stale package paths, redirect modules, Any
annotations, and the required package layout of the target tree. It prints a
concise table; exit code 0 means no actionable findings.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "fedcrg"

_FORBIDDEN_FRAGMENTS = (
    "canonical",
    "TODO",
    "FIXME",
    "Placeholder",
    "D:\\Projects\\FedCRG",
)
_STALE_PACKAGES = (
    "configuration",
    "datasets",
    "decision",
    "evaluation",
    "artifacts",
    "analysis",
    "domain",
    "federation",
    "scoring",
    "detectors",
    "cli",
    "runtime",
    "pipeline",
    "method",
    "thresholds",
)
_REQUIRED_TOP_LEVEL = (
    "config.py",
    "types.py",
    "runtime.py",
    "reporting.py",
    "cli.py",
)
_REQUIRED_PACKAGES = ("data", "learning", "thresholding", "evidence", "experiments")
_VAGUE_NAMES = (
    "utils",
    "helpers",
    "common",
    "manager",
    "handler",
    "processor",
    "engine",
    "service",
    "base",
    "models",
    "registry",
    "factory",
)
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

    for name in _STALE_PACKAGES:
        if (SRC / name).exists():
            findings.append(f"src/fedcrg/{name}/ still exists (stale package)")

    for name in _REQUIRED_TOP_LEVEL:
        if not (SRC / name).is_file():
            findings.append(f"src/fedcrg/{name} missing")

    for package in _REQUIRED_PACKAGES:
        if not (SRC / package).is_dir():
            findings.append(f"src/fedcrg/{package}/ missing")

    for name in _VAGUE_NAMES:
        if (SRC / f"{name}.py").is_file():
            findings.append(f"src/fedcrg/{name}.py: vague filename {name}")

    for config_name in ("study.yaml", "datasets.yaml", "experiments.yaml"):
        if not (ROOT / "config" / config_name).is_file():
            findings.append(f"config/{config_name} missing")

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
