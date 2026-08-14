"""Terminology and comment hygiene contract (goal §3, §6, §37).

Production source must not reference the roadmap, the audit matrix, this
implementation goal, migration, legacy, compatibility, or old architecture.
The word ``canonical`` and its derivatives are forbidden. Comments must
explain only useful scientific or engineering rationale; they must not
narrate syntax or implementation history.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fedcrg"

_FORBIDDEN_TERMS = (
    "canonical",
    "canonicalize",
    "canonicalized",
    "canonical_name",
    "canonical_payload",
    "roadmap",
    "audit matrix",
    "prompt",
    "migration",
    "legacy",
    "compatibility",
    "old implementation",
    "old architecture",
    "previous structure",
    "backward compat",
    "backwards compat",
    "compat shim",
)

_AI_COMMENT_PATTERNS = (
    re.compile(r"\b(note that|simply|basically|essentially|in other words|let'?s)\b"),
    re.compile(r"\b(placeholder|stub|temporary workaround|hack|TODO|FIXME|XXX)\b"),
    re.compile(r"\b(we |our |us |I )\b"),
)


def _python_files() -> tuple[Path, ...]:
    return tuple(sorted(SRC.rglob("*.py")))


def test_no_forbidden_terminology_in_production_source() -> None:
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            lowered = line.lower()
            for term in _FORBIDDEN_TERMS:
                if term in lowered:
                    violations.append(f"{relative}:{line_number}: {term!r}")
    assert not violations, "Forbidden terminology remains:\n" + "\n".join(violations)


def test_no_implementation_tracking_phase_numbers() -> None:
    """Phase numbers used only for implementation tracking must not appear."""
    pattern = re.compile(r"\bP\d+\b")
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                violations.append(f"{relative}:{line_number}: {line.strip()}")
    assert not violations, "Implementation phase numbers remain:\n" + "\n".join(violations)


def test_comments_are_rationale_not_narration() -> None:
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            docstring = ast.get_docstring(node)
            if docstring is None:
                continue
            lowered = docstring.lower()
            for pattern in _AI_COMMENT_PATTERNS:
                match = pattern.search(lowered)
                if match and match.group(1) not in {"i ", "us "}:
                    violations.append(
                        f"{relative}:{node.lineno} {node.name}: docstring contains "
                        f"{match.group(1)!r}"
                    )
    assert not violations, "AI-style commentary remains:\n" + "\n".join(violations)


def test_no_implementation_history_in_comments() -> None:
    history_terms = (
        "was previously",
        "used to be",
        "was formerly",
        "moved from",
        "renamed from",
        "replaced the old",
        "removed the old",
        "kept for",
    )
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            docstring = ast.get_docstring(node) or ""
            lowered = docstring.lower()
            for term in history_terms:
                if term in lowered:
                    violations.append(f"{relative}:{node.lineno} {node.name}: {term!r}")
    assert not violations, "Implementation-history commentary remains:\n" + "\n".join(
        violations
    )


def test_public_docstrings_describe_contracts() -> None:
    """Public classes and functions must carry a docstring describing semantics."""
    missing: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                if ast.get_docstring(node) is None:
                    missing.append(f"{relative}:{node.lineno} {node.name}")
    assert not missing, "Public definitions lack docstrings:\n" + "\n".join(missing)
