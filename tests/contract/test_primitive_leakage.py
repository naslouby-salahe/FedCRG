from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fedcrg"

_PRIMITIVE_LEAF = {"float", "int", "str", "object", "Any"}

_ALLOWED_ANNOTATIONS: dict[tuple[str, str, str], str] = {
    ("config.py", "load_yaml_mapping", "return"): "YAML parse boundary before validation",
    ("evidence/store.py", "atomic_write_text", "content"): "serialization output is str",
    ("evidence/store.py", "atomic_write_json", "payload"): "arbitrary JSON payload boundary",
    ("evidence/store.py", "_jsonable", "value"): "JSON serialization boundary",
    ("evidence/store.py", "load_yaml_mapping", "return"): "YAML parse boundary before validation",
    (
        "evidence/store.py",
        "run",
        "args",
    ): "subprocess argv boundary (git capture passes raw strings)",
    (
        "evidence/store.py",
        "run",
        "return",
    ): "subprocess stdout boundary (git capture returns str)",
    ("runtime.py", "log_stage", "fields"): "structured-logging kwargs boundary",
    (
        "reporting.py",
        "arrow",
        "start",
    ): "matplotlib figure coordinate boundary",
    (
        "reporting.py",
        "arrow",
        "end",
    ): "matplotlib figure coordinate boundary",
    (
        "reporting.py",
        "arrow",
        "label",
    ): "matplotlib figure text boundary",
}

_COERCE_VALIDATORS = (
    "_coerce_client_keys",
    "_coerce_keys",
    "_coerce_values",
    "_coerce_settings",
    "_coerce_axes",
    "_coerce_datasets",
    "_coerce_required_evidence",
    "_coerce_root",
    "_coerce_calibration_seeds",
    "_coerce_device_directories",
)
for _name in _COERCE_VALIDATORS:
    _ALLOWED_ANNOTATIONS[("config.py", _name, "return")] = "YAML/JSON input before validation"
    _ALLOWED_ANNOTATIONS[("config.py", _name, "value")] = "YAML/JSON input before validation"

_ALLOWED_ANNOTATIONS[("config.py", "serialized_payload", "return")] = "JSON serialization output"
_ALLOWED_ANNOTATIONS[("hashing.py", "sha256_text", "text")] = (
    "UTF-8 text digest input (crypto boundary)"
)
_ALLOWED_ANNOTATIONS[("data/datasets.py", "hash_row_ids", "values")] = (
    "row-id collection boundary (RowId | str)"
)
_ALLOWED_ANNOTATIONS[("data/datasets.py", "hash_seed", "text")] = (
    "hash input string (crypto boundary)"
)
_ALLOWED_ANNOTATIONS[("data/datasets.py", "stable_row_id", "source")] = (
    "source-file path string (filesystem boundary)"
)
_ALLOWED_ANNOTATIONS[("data/splits.py", "validate_split_disjointness", "row_id_column")] = (
    "pandas column-name boundary"
)
_ALLOWED_ANNOTATIONS[("data/diad.py", "_normalized_mac", "value")] = (
    "raw dataset text boundary before MAC normalization"
)


def _python_files() -> tuple[Path, ...]:
    return tuple(sorted(SRC.rglob("*.py")))


def _annotation_leaves(annotation: ast.expr) -> set[str]:
    leaves: set[str] = set()
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id in _PRIMITIVE_LEAF:
            leaves.add(node.id)
    return leaves


def _is_dunder_len(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return node.name == "__len__"


def test_no_any_in_production_source() -> None:
    violations: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "Any":
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            if isinstance(node, ast.Attribute) and node.attr == "Any":
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, "Any remains in production code:\n" + "\n".join(violations)


def test_no_weak_generic_mappings() -> None:
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if node.returns is not None and _annotation_is_weak(node.returns):
                    key = (relative, node.name, "return")
                    if key not in _ALLOWED_ANNOTATIONS:
                        violations.append(
                            f"{relative}:{node.lineno}: {node.name} returns "
                            f"{ast.unparse(node.returns)}"
                        )
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if isinstance(
                    _enclosing_scope(tree, node), (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                if node.annotation is not None and _annotation_is_weak(node.annotation):
                    key = (relative, "", node.target.id)
                    if key not in _ALLOWED_ANNOTATIONS:
                        violations.append(
                            f"{relative}:{node.lineno}: field {node.target.id} is "
                            f"{ast.unparse(node.annotation)}"
                        )
    assert not violations, "Weak generic mappings remain:\n" + "\n".join(violations)


def _enclosing_scope(tree: ast.AST, node: ast.AST) -> ast.AST | None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if child is node:
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    return parent
                return _enclosing_scope(tree, parent)
    return None


def _annotation_is_weak(annotation: ast.expr) -> bool:
    source = ast.unparse(annotation)
    if source in {"object", "Any"}:
        return True
    if source.startswith("dict[") or source.startswith("Mapping["):
        inner = source[source.find("[") + 1 : -1]
        if inner.startswith("str, object") or inner.startswith("str, Any"):
            return True
    if "list[dict[" in source:
        return True
    return False


def _is_click_handler(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        name = ""
        if isinstance(decorator, ast.Call):
            decorator = decorator.func
        if isinstance(decorator, ast.Name):
            name = decorator.id
        elif isinstance(decorator, ast.Attribute):
            name = decorator.attr
        if name in {"command", "group"} or name.endswith("_command"):
            return True
    return False


def _is_json_parse(value: ast.expr | None) -> bool:
    if not isinstance(value, ast.Call):
        return False
    function = value.func
    if isinstance(function, ast.Attribute):
        return function.attr in {"loads", "safe_load", "load"}
    if isinstance(function, ast.Name):
        return function.id in {"loads", "safe_load", "load"}
    return False


def _collect_annotation_violations() -> list[str]:
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        def visit(
            node: ast.AST,
            owner: str,
            in_function: bool = False,
            relative: str = relative,
        ) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = node.name
                if node.returns is not None:
                    for leaf in sorted(_annotation_leaves(node.returns)):
                        key = (relative, scope, "return")
                        if key not in _ALLOWED_ANNOTATIONS and not (
                            leaf == "int" and _is_dunder_len(node)
                        ):
                            violations.append(f"{relative}:{node.lineno} {scope} returns {leaf}")
                cli_input = _is_click_handler(node)
                parameters = [
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                ]
                if node.args.vararg is not None:
                    parameters.append(node.args.vararg)
                if node.args.kwarg is not None:
                    parameters.append(node.args.kwarg)
                for arg in parameters:
                    if arg.annotation is None:
                        continue
                    for leaf in sorted(_annotation_leaves(arg.annotation)):
                        key = (relative, scope, arg.arg)
                        if key not in _ALLOWED_ANNOTATIONS and not cli_input:
                            violations.append(
                                f"{relative}:{node.lineno} {scope} parameter {arg.arg}: {leaf}"
                            )
                for child in node.body:
                    visit(child, scope, True)
                return
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    visit(child, node.name, in_function)
                return
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                field = node.target.id
                if node.annotation is None:
                    return
                leaves = sorted(_annotation_leaves(node.annotation))
                if in_function:
                    if leaves == ["object"] and _is_json_parse(node.value):
                        return
                    for leaf in leaves:
                        violations.append(f"{relative}:{node.lineno} local {field}: {leaf}")
                    return
                for leaf in leaves:
                    key = (relative, owner, field)
                    if key not in _ALLOWED_ANNOTATIONS:
                        violations.append(f"{relative}:{node.lineno} field {field}: {leaf}")

        for child in tree.body:
            visit(child, "", False)

    return violations


def test_no_primitive_leaks_outside_approved_boundaries() -> None:
    violations = _collect_annotation_violations()
    assert not violations, "Primitive leaks outside approved boundaries:\n" + "\n".join(violations)


def test_no_bare_dict_or_list_annotations() -> None:
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            annotation: ast.expr | None = None
            if isinstance(node, ast.AnnAssign):
                annotation = node.annotation
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotation = node.returns
            if annotation is None:
                continue
            location = getattr(node, "lineno", 0)
            subscripted: set[int] = set()
            for child in ast.walk(annotation):
                if isinstance(child, ast.Subscript):
                    subscripted.add(id(child.value))
            for child in ast.walk(annotation):
                if isinstance(child, ast.Name) and child.id in ("dict", "list", "Mapping"):
                    if id(child) not in subscripted:
                        violations.append(f"{relative}:{location}: bare {child.id}")
    assert not violations, "Bare dict/list/Mapping annotations remain:\n" + "\n".join(violations)
