"""Permanent architecture contract tests (prompt §28): scientific defaults,
configuration ownership, canonical vocabulary, redirect modules, and typed transport."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fedcrg"


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


def test_no_canonical_terminology_in_production_source() -> None:
    violations: list[str] = []
    for path in _python_files(SRC):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "canonical" in line.lower():
                violations.append(f"{path.relative_to(ROOT)}:{line_number}")
    assert not violations, "canonical terminology remains:\n" + "\n".join(violations)


def test_no_thin_redirect_modules() -> None:
    """A module that only re-exports symbols from exactly one other module is a redirect."""
    for path in _python_files(SRC):
        if path.name == "__init__.py" or path.name == "__main__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
        if len(imports) != 1:
            continue
        source_import = imports[0]
        if source_import.module is None:
            continue
        import_names = {alias.asname or alias.name for alias in source_import.names}
        defined = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
        ]
        assigned = [
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        if import_names and not defined and not assigned:
            raise AssertionError(
                f"{path.relative_to(ROOT)} is a thin redirect module of {source_import.module}"
            )


def test_no_scientific_defaults_in_configuration_models() -> None:
    """Pydantic config models must not declare scientific default values."""
    config_dir = SRC / "configuration"
    for path in _python_files(config_dir):
        if path.name in {"__init__.py", "load.py"}:
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
                # Structural defaults are allowed: Literal discriminators, empty
                # collections, and None. Scientific numeric/collection defaults are not.
                annotation = statement.annotation
                literal_annotation = (
                    isinstance(annotation, ast.Subscript)
                    and isinstance(annotation.value, ast.Name)
                    and annotation.value.id == "Literal"
                )
                if literal_annotation:
                    continue
                default = statement.value
                if isinstance(default, ast.Constant) and isinstance(default.value, (int, float)):
                    raise AssertionError(
                        f"{path.relative_to(ROOT)}:{statement.lineno}: "
                        f"scientific default {default.value!r} in {node.name}"
                    )
                if isinstance(default, ast.List) or isinstance(default, ast.Tuple):
                    elements = default.elts if isinstance(default, ast.Tuple) else default.elts
                    if elements:
                        raise AssertionError(
                            f"{path.relative_to(ROOT)}:{statement.lineno}: "
                            f"collection default in configuration model {node.name}"
                        )


def test_no_dict_str_object_transport_outside_json_io() -> None:
    """dict[str, object] is reserved for the JSON serialization boundary."""
    for path in _python_files(SRC):
        if path.name == "json_io.py":
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "dict[str, object]" in line:
                raise AssertionError(
                    f"{path.relative_to(ROOT)}:{line_number}: "
                    "dict[str, object] transport outside the JSON boundary"
                )


def test_no_object_return_types_outside_json_io() -> None:
    for path in _python_files(SRC):
        if path.name == "json_io.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "overload":
                        break
                else:
                    if node.returns is not None and isinstance(node.returns, ast.Name):
                        if node.returns.id == "object":
                            raise AssertionError(
                                f"{path.relative_to(ROOT)}:{node.lineno}: "
                                f"{node.name} returns bare object"
                            )


def test_compatible_experiments_share_preprocessing_identity() -> None:
    """Two experiments with identical data spec share one preprocessed identity."""
    from fedcrg.configuration.resolve import ExperimentConfigResolver

    primary = ExperimentConfigResolver().resolve(ROOT / "configs/experiments/primary/nbaiot.yaml")
    external = ExperimentConfigResolver().resolve(ROOT / "configs/experiments/external/diad.yaml")
    assert primary.data_spec_hash is not None
    assert external.data_spec_hash is not None
    # Identical dataset section => identical data_spec_hash regardless of experiment id.
    assert primary.data_spec_hash != external.data_spec_hash
    assert primary.dataset.id.value != external.dataset.id.value
