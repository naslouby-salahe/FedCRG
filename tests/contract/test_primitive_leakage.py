"""AST-based primitive-leakage contract (goal §7, §8).

Scans every production annotation and flags raw ``float``/``int``/``str``/
``object``/``Any``/bare ``dict``/bare ``list``/weak generic mappings outside an
explicit, architecturally justified boundary allowlist.

Typed generics (``dict[ClientId, PositiveCount]``, ``tuple[PolicyId, ...]``)
are strong types and are never flagged; only bare unsubscripted containers and
raw scalar/object/Any leaves are violations.

The allowlist is documented by architectural reason, not by convenience:
- ``np.ndarray``/``torch.Tensor`` are library boundaries; arrays are legitimate
  primitive carriers and are never flagged.
- YAML/JSON parse functions accept and return ``object`` before validation;
  that is the JSON/YAML boundary.
- ``__len__`` returns ``int`` because the Python data model requires it.
- Free-text metadata fields (descriptions, version strings) have no meaningful
  constrained type.
- Atomic serialization helpers produce ``str`` output by definition.
- Internal low-level arithmetic inside a scientifically clear implementation
  may use float64 scalars.

Any ``Any`` usage anywhere is a hard failure and cannot be allowlisted.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fedcrg"

_PRIMITIVE_LEAF = {"float", "int", "str", "object", "Any"}

# (relative path, enclosing class or function, member name) -> architectural reason.
_ALLOWED_ANNOTATIONS: dict[tuple[str, str, str], str] = {
    ("config.py", "load_yaml_mapping", "return"): "YAML parse boundary before validation",
    ("config.py", "_sha256_json", "payload"): "generic JSON payload hash boundary",
    ("evidence/store.py", "atomic_write_text", "content"): "serialization output is str",
    ("evidence/store.py", "atomic_write_json", "payload"): "arbitrary JSON payload boundary",
    ("evidence/store.py", "_jsonable", "value"): "JSON serialization boundary",
    ("evidence/store.py", "load_yaml_mapping", "return"): "YAML parse boundary before validation",
}

# Pydantic before-validators receive and return unvalidated YAML/JSON input.
_COERCE_VALIDATORS = (
    "_coerce_client_keys",
    "_coerce_keys",
    "_coerce_values",
    "_coerce_settings",
    "_coerce_axes",
    "_coerce_datasets",
    "_coerce_required_evidence",
    "_coerce_root",
)
for _name in _COERCE_VALIDATORS:
    _ALLOWED_ANNOTATIONS[("config.py", _name, "return")] = "YAML/JSON input before validation"
    _ALLOWED_ANNOTATIONS[("config.py", _name, "value")] = "YAML/JSON input before validation"

_ALLOWED_ANNOTATIONS[("config.py", "serialized_payload", "return")] = "JSON serialization output"
_ALLOWED_ANNOTATIONS[("data/datasets.py", "hash_file", "chunk_size")] = "filesystem read chunk boundary"
_ALLOWED_ANNOTATIONS[("data/datasets.py", "hash_row_ids", "values")] = "row-id collection boundary (RowId | str)"
_ALLOWED_ANNOTATIONS[("data/datasets.py", "hash_seed", "text")] = "hash input string (crypto boundary)"
_ALLOWED_ANNOTATIONS[("data/datasets.py", "stable_row_id", "source")] = "source-file path string (filesystem boundary)"
_ALLOWED_ANNOTATIONS[("data/datasets.py", "validate_split_disjointness", "row_id_column")] = "pandas column-name boundary"
_ALLOWED_ANNOTATIONS[("data/datasets.py", "_ensure_row_ids", "source")] = "source-file path string (filesystem boundary)"
_ALLOWED_ANNOTATIONS[("data/datasets.py", "_normalized_name", "return")] = "string normalization helper output"
_ALLOWED_ANNOTATIONS[("data/datasets.py", "_normalized_name", "path")] = "filesystem path boundary" 

# No free-text exception: every model field must carry a meaningful constrained
# type (Version, Description, Identifier, Sha256, ...) rather than a bare str.


def _python_files() -> tuple[Path, ...]:
    return tuple(sorted(SRC.rglob("*.py")))


def _annotation_leaves(annotation: ast.expr) -> set[str]:
    """Collect bare primitive names referenced inside one annotation.

    A raw ``str`` used as a dict/Mapping key is still a primitive leak; the key
    must be a constrained identifier type, not ``str``. Weak generic values
    (``object``/``Any``) are caught by ``test_no_weak_generic_mappings``.
    """
    leaves: set[str] = set()
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id in _PRIMITIVE_LEAF:
            leaves.add(node.id)
    return leaves


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


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
        source = path.read_text(encoding="utf-8")
        for fragment in (
            "dict[str, object]",
            "dict[str, Any]",
            "Mapping[str, object]",
            "Mapping[str, Any]",
            "list[dict[",
            "-> object",
            "-> Any",
        ):
            for line_number, line in enumerate(source.splitlines(), 1):
                if fragment in line:
                    violations.append(f"{path.relative_to(ROOT)}:{line_number}: {fragment}")
    assert not violations, "Weak generic mappings remain:\n" + "\n".join(violations)


def _is_click_handler(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the function is a click command handler.

    CLI input arrives before validation; click passes raw strings/ints to the
    handler, which converts them to typed values immediately (goal §7 boundary).
    """
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


def _collect_annotation_violations() -> list[str]:
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        def visit(node: ast.AST, owner: str, in_function: bool = False) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = node.name
                if node.returns is not None:
                    for leaf in sorted(_annotation_leaves(node.returns)):
                        key = (relative, scope, "return")
                        if key not in _ALLOWED_ANNOTATIONS and not (
                            leaf == "int" and _is_dunder_len(node)
                        ):
                            violations.append(
                                f"{relative}:{node.lineno} {scope} returns {leaf}"
                            )
                cli_input = _is_click_handler(node)
                for arg in node.args.args:
                    if arg.annotation is None:
                        continue
                    for leaf in sorted(_annotation_leaves(arg.annotation)):
                        key = (relative, scope, arg.arg)
                        if key not in _ALLOWED_ANNOTATIONS and not cli_input:
                            violations.append(
                                f"{relative}:{node.lineno} {scope} parameter {arg.arg}: {leaf}"
                            )
                return
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    visit(child, node.name, in_function)
                return
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if in_function:
                    return
                field = node.target.id
                if node.annotation is None:
                    return
                for leaf in sorted(_annotation_leaves(node.annotation)):
                    key = (relative, owner, field)
                    if key not in _ALLOWED_ANNOTATIONS:
                        violations.append(f"{relative}:{node.lineno} field {field}: {leaf}")

        for child in tree.body:
            visit(child, "", False)

    return violations


def test_no_primitive_leaks_outside_approved_boundaries() -> None:
    violations = _collect_annotation_violations()
    assert not violations, "Primitive leaks outside approved boundaries:\n" + "\n".join(
        violations
    )


def test_no_bare_dict_or_list_annotations() -> None:
    """Only unsubscripted ``dict``/``list``/``Mapping`` leaves are bare containers.

    ``dict[ClientId, PositiveCount]`` is a strong typed mapping, not a leak.
    """
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
