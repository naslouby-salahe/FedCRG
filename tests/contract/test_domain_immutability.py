from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fedcrg"

_OUTER_BOUNDARIES = {
    "artifacts/json_io.py",
    "artifacts/records.py",
}
_MUTABLE_NAMES = {"dict", "list", "set", "Dict", "List", "Set"}


def _is_frozen_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        target = decorator.func
        name = (
            target.id
            if isinstance(target, ast.Name)
            else target.attr
            if isinstance(target, ast.Attribute)
            else ""
        )
        if name != "dataclass":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "frozen" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value is True
    return False


def _annotation_uses_mutable_container(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    return any(
        isinstance(node, ast.Name) and node.id in _MUTABLE_NAMES for node in ast.walk(annotation)
    )


def test_frozen_domain_models_do_not_embed_mutable_containers() -> None:
    violations: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        if relative in _OUTER_BOUNDARIES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or not _is_frozen_dataclass(node):
                continue
            for statement in node.body:
                if isinstance(statement, ast.AnnAssign) and _annotation_uses_mutable_container(
                    statement.annotation
                ):
                    violations.append(
                        f"{path.relative_to(ROOT)}:{statement.lineno} "
                        f"{node.name}.{getattr(statement.target, 'id', '?')}"
                    )
    assert not violations, (
        "Frozen scientific/domain models must use immutable tuple/value records, not "
        "mutable containers:\n" + "\n".join(violations)
    )
