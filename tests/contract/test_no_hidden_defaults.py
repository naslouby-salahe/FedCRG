"""No-hidden-scientific-defaults contract.

Scientific configuration values must not silently appear because Python
supplied a default: function argument defaults, dataclass defaults, Pydantic
defaults, CLI defaults, ``.get(..., default)`` fallbacks, and ``or <value>``
fallback logic are audited. Structural/runtime conveniences (``None``,
empty collections, boolean flags, output paths) are allowed because they
cannot alter scientific interpretation.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fedcrg"

# Module/class names that legitimately own scientific constants by design.
_ALLOWED_OWNERS = {
    "types.py",
    "config.py",
}

# Parameter names that are structural, not scientific.
_STRUCTURAL_PARAMS = {
    "outputs_root",
    "preprocessed_root",
    "repository_root",
    "prepared_root",
    "data_root",
    "path",
    "output",
    "seed",  # seeds are configured elsewhere; a raw int default would be caught
}


def _python_files() -> tuple[Path, ...]:
    return tuple(sorted(SRC.rglob("*.py")))


def test_no_scientific_function_argument_defaults() -> None:
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        if relative in _ALLOWED_OWNERS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for default in node.args.defaults:
                if _is_scientific_default(default):
                    violations.append(f"{relative}:{node.lineno} {node.name} default")
            for default in node.args.kw_defaults:
                if default is not None and _is_scientific_default(default):
                    violations.append(f"{relative}:{node.lineno} {node.name} keyword default")
    assert not violations, "Scientific function defaults remain:\n" + "\n".join(violations)


def _is_scientific_default(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or node.value is None:
            return False
        if isinstance(node.value, (int, float)):
            return node.value != 0
        return False
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and node.elts:
        return True
    return False


def test_no_or_fallback_scientific_values() -> None:
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
                continue
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
                    if value.value != 0:
                        violations.append(f"{relative}:{node.lineno} or-fallback {value.value!r}")
    assert not violations, "or-fallback scientific values remain:\n" + "\n".join(violations)


def test_no_get_fallback_scientific_values() -> None:
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (isinstance(function, ast.Attribute) and function.attr == "get"):
                continue
            if len(node.args) < 2:
                continue
            fallback = node.args[1]
            if isinstance(fallback, ast.Constant) and isinstance(fallback.value, (int, float)):
                if fallback.value != 0:
                    violations.append(f"{relative}:{node.lineno} .get fallback {fallback.value!r}")
    assert not violations, ".get scientific fallbacks remain:\n" + "\n".join(violations)


def test_no_scientific_pydantic_defaults_outside_config_owner() -> None:
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        if relative == "config.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                if not isinstance(statement, ast.AnnAssign):
                    continue
                if statement.value is None:
                    continue
                if isinstance(statement.value, ast.Constant) and isinstance(
                    statement.value.value, (int, float)
                ):
                    if statement.value.value != 0:
                        target = statement.target
                        name = target.id if isinstance(target, ast.Name) else "?"
                        violations.append(f"{relative}:{statement.lineno} {node.name}.{name}")
    assert not violations, "Scientific Pydantic defaults remain:\n" + "\n".join(violations)
