"""Config-vs-source drift contract (goal §11, §12).

Configured scientific values must not be duplicated as literals in production
source. The test parses the three configuration documents, collects configured
scientific values, scans production AST literals, and reports any configured
value repeated directly in source.

Mathematical constants that are inherent to an algorithm are the only
allowlist entries and each is documented by scientific reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fedcrg"

# (relative path, value, reason) — inherent mathematical constants only.
_ALLOWED_SOURCE_LITERALS: dict[tuple[str, object], str] = {
    ("types.py", 1.0e-12): "tight comparison tolerance for exact numeric contracts",
    ("types.py", 1.0e-10): "exact beta-function tolerance locked by the protocol",
    ("types.py", 1.0e-9): "coarse tolerance for regenerated tables",
    ("config.py", 1.0e-12): "ranking-invariance tolerance owned by configuration",
}


def _configured_values() -> set[object]:
    values: set[object] = set()
    for name in ("study.yaml", "datasets.yaml", "experiments.yaml"):
        raw = yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for key, item in node.items():
                    if key in {"id", "name", "description", "version", "source_version",
                               "parser_version", "feature_contract"}:
                        continue
                    walk(item)
            elif isinstance(node, list):
                for item in node:
                    walk(item)
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                values.add(node)
            elif isinstance(node, str):
                try:
                    values.add(float(node))
                except ValueError:
                    pass

        walk(raw)
    return values


def _source_literals() -> dict[tuple[str, object], list[int]]:
    occurrences: dict[tuple[str, object], list[int]] = {}
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                value = float(node.value)
                key = (relative, value)
                occurrences.setdefault(key, []).append(getattr(node, "lineno", 0))
    return occurrences


def test_configured_values_not_duplicated_in_source() -> None:
    configured = _configured_values()
    occurrences = _source_literals()
    violations: list[str] = []
    for (relative, value), lines in sorted(occurrences.items(), key=str):
        if value not in configured:
            continue
        if (relative, value) in _ALLOWED_SOURCE_LITERALS:
            continue
        for line in lines[:3]:
            violations.append(f"{relative}:{line} repeats configured value {value!r}")
    assert not violations, (
        "Configured scientific values duplicated in production source:\n"
        + "\n".join(violations)
    )


def test_experiment_axes_not_duplicated_in_python() -> None:
    raw = yaml.safe_load((ROOT / "config" / "experiments.yaml").read_text(encoding="utf-8"))
    axis_values: set[object] = set()
    for experiment in raw["experiments"]:
        for axis, values in (experiment.get("axes") or {}).items():
            if isinstance(values, list):
                axis_values.update(values)
        for cell in experiment.get("coupled_cells") or []:
            axis_values.update(cell.values())
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if isinstance(node.value, (int, float)) and float(node.value) in {
                float(item) for item in axis_values if isinstance(item, (int, float))
            }:
                raise AssertionError(
                    f"{relative}:{getattr(node, 'lineno', 0)} duplicates experiment axis "
                    f"value {node.value!r}"
                )


def test_configured_seed_lists_not_duplicated_in_python() -> None:
    raw = yaml.safe_load((ROOT / "config" / "datasets.yaml").read_text(encoding="utf-8"))
    seeds: set[int] = set()
    for contract in raw["datasets"].values():
        seeds.update(contract.get("calibration_seeds") or [])
    study = yaml.safe_load((ROOT / "config" / "study.yaml").read_text(encoding="utf-8"))
    seeds.update(study["randomness"]["model_seeds"])
    seeds.add(study["randomness"]["attack_split_seed"])
    seeds.add(study["randomness"]["synthetic_seed"])
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                if node.value in seeds:
                    raise AssertionError(
                        f"{relative}:{getattr(node, 'lineno', 0)} duplicates configured "
                        f"seed {node.value!r}"
                    )
