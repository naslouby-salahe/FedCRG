from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fedcrg"

_FORBIDDEN_SOURCE_FRAGMENTS = (
    "model_copy(update=",
    "__import__(",
    "TODO",
    "FIXME",
    "Placeholder",
    "/home/naslouby/Projects/FedCRG",
    "D:\\Projects\\FedCRG",
)


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_legacy_repository_layers_are_absent() -> None:
    assert not (ROOT / "scripts").exists()
    assert not (ROOT / "fedcrg").exists()
    assert not (ROOT / "setup.py").exists()


def test_legacy_protocol_shorthand_modules_are_absent() -> None:
    forbidden = {"gate_a.py", "gate_b.py"}
    found = {path.name for path in _python_files(SRC)} & forbidden
    assert not found


def test_no_any_annotations_or_imports_in_package() -> None:
    violations: list[str] = []
    for path in _python_files(SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "Any":
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if isinstance(node, ast.Attribute) and node.attr == "Any":
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, "Any remains in production code:\n" + "\n".join(violations)


def test_no_forbidden_implementation_shortcuts() -> None:
    violations: list[str] = []
    for path in _python_files(SRC):
        source = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_SOURCE_FRAGMENTS:
            if fragment in source:
                violations.append(f"{path.relative_to(ROOT)} contains {fragment!r}")
    assert not violations, "\n".join(violations)


def test_domain_is_dependency_free_from_outer_layers() -> None:
    forbidden_prefixes = (
        "fedcrg.application",
        "fedcrg.artifacts",
        "fedcrg.cli",
        "fedcrg.config",
        "fedcrg.data",
        "fedcrg.detectors",
        "fedcrg.experiments",
        "fedcrg.federated",
        "fedcrg.metrics",
        "fedcrg.policies",
        "fedcrg.protocol",
        "fedcrg.scoring",
    )
    violations: list[str] = []
    for path in _python_files(SRC / "domain"):
        for imported in _imports(path):
            if imported.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(ROOT)} -> {imported}")
    assert not violations, "\n".join(violations)


def test_protocol_does_not_depend_on_application_cli_or_artifact_io() -> None:
    forbidden_prefixes = (
        "fedcrg.application",
        "fedcrg.cli",
        "fedcrg.experiments",
    )
    violations: list[str] = []
    for path in _python_files(SRC / "protocol"):
        for imported in _imports(path):
            if imported.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(ROOT)} -> {imported}")
    assert not violations, "\n".join(violations)


def test_cli_never_imports_statistical_or_training_implementation_directly() -> None:
    forbidden_prefixes = (
        "fedcrg.protocol",
        "fedcrg.federated",
        "fedcrg.detectors",
        "fedcrg.policies",
        "fedcrg.metrics",
    )
    violations: list[str] = []
    for path in _python_files(SRC / "cli"):
        for imported in _imports(path):
            if imported.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(ROOT)} -> {imported}")
    assert not violations, "CLI must call application services only:\n" + "\n".join(violations)
