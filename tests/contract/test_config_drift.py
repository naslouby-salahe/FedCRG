from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fedcrg"

_ALLOWED_SOURCE_LITERALS: dict[tuple[str, object], str] = {
    ("types.py", 1.0e-12): "tight comparison tolerance for exact numeric contracts",
    ("types.py", 1.0e-10): "exact beta-function tolerance locked by the protocol",
    ("types.py", 1.0e-9): "coarse tolerance for regenerated tables",
    ("config.py", 1.0e-12): "ranking-invariance tolerance owned by configuration",
    ("data/datasets.py", 0.99): "minimum finite-rate for DIAD eligibility evaluation",
    ("data/diad.py", 0.75): "encoder depth schedule: 3/4 of input dimension",
    ("data/diad.py", 0.5): "encoder depth schedule: half of input dimension",
    ("data/diad.py", 0.25): "encoder depth schedule: quarter of input dimension",
    ("data/diad.py", 3): "encoder depth schedule: one third of input dimension",
    ("data/preprocessing.py", 0.99): "minimum finite-rate for numeric-safe feature derivation",
    ("evidence/store.py", 1024): "filesystem hashing chunk size in KiB (IO boundary)",
    ("evidence/store.py", 1_000_000): "run-id alpha encoding in parts per million",
    ("evidence/store.py", 10_000): "run-id rho/assurance/confidence encoding in basis points",
    ("experiments/analyses.py", 0.005): "Monte-Carlo acceptance floor for coverage validation",
    (
        "experiments/analyses.py",
        0.1,
    ): "normal-mixture synthetic distribution: 10% contaminated component",
    ("experiments/analyses.py", 0.9): "normal-mixture synthetic distribution: 90% clean component",
    ("experiments/analyses.py", 1000): "seed offset scale converting float axis values to integers",
    (
        "experiments/analyses.py",
        3.0,
    ): "normal-mixture and contamination shift location in standard deviations",
    ("experiments/analyses.py", 5): "source-order analysis block count and 5th-percentile quantile",
    ("experiments/analyses.py", 25): "IQR lower quantile (25th percentile)",
    ("experiments/analyses.py", 75): "IQR upper quantile (75th percentile)",
    ("experiments/analyses.py", 95): "upper percentile quantile (95th percentile)",
    ("learning/federated.py", 0.5): "cosine learning-rate schedule midpoint",
    ("learning/scores.py", 1024): "filesystem hashing chunk size in KiB (IO boundary)",
    ("learning/scores.py", 64): "SHA-256 hexadecimal digest length",
    ("reporting.py", 0.5): "decision-figure canvas coordinate",
    ("reporting.py", 0.6): "decision-figure canvas coordinate",
    ("reporting.py", 10.0): "decision-figure canvas coordinate",
    ("reporting.py", 300): "figure rasterization dpi",
    ("reporting.py", 5.0): "decision-figure canvas coordinate",
    ("reporting.py", 9.0): "decision-figure canvas coordinate and label fontsize",
    ("thresholding/metrics.py", 0.5): "balanced-accuracy mean of sensitivity and specificity",
    ("thresholding/policies.py", 3.0): "three-sigma threshold standard-deviation multiple",
    ("thresholding/policies.py", 500): "supervised development budget: 500 benign + 500 malicious",
    ("types.py", 100.0): "percentage bound upper limit",
    ("types.py", 64): "SHA-256 hexadecimal digest length",
}


def _configured_values() -> set[object]:
    values: set[object] = set()
    for name in ("study.yaml", "datasets.yaml", "experiments.yaml"):
        raw = yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for key, item in node.items():
                    if key in {
                        "id",
                        "name",
                        "description",
                        "version",
                        "source_version",
                        "parser_version",
                        "feature_contract",
                    }:
                        continue
                    walk(item)
            elif isinstance(node, list):
                for item in node:
                    walk(item)
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                if node in (0, 0.0, 1, 1.0):
                    return
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
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                continue
            parent = _parent_of(tree, node)
            if _is_structural_context(parent, node):
                continue
            if _is_presentation_context(tree, node):
                continue
            value = float(node.value)
            key = (relative, value)
            occurrences.setdefault(key, []).append(getattr(node, "lineno", 0))
    return occurrences


def _parent_of(tree: ast.AST, node: ast.AST) -> ast.AST | None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if child is node:
                return parent
    return None


def _is_structural_context(parent: ast.AST | None, node: ast.Constant) -> bool:
    if parent is None:
        return False
    if isinstance(parent, ast.Slice):
        return True
    if isinstance(parent, ast.Subscript) and parent.slice is node:
        return True
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name):
        if parent.func.id == "range":
            return True
        if parent.func.id in {"round", "int", "float", "max", "min", "abs", "len"}:
            return True
        if parent.func.id in {"SystemExit", "exit"}:
            return True
        if parent.func.id.endswith("get_device_name") or parent.func.id == "set_device":
            return True
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute):
        if parent.func.attr.endswith("get_device_name") or parent.func.attr == "set_device":
            return True
    if isinstance(parent, ast.keyword) and parent.arg in {
        "timeout",
        "workers",
        "chunk_size",
        "default",
    }:
        return True
    if isinstance(parent, ast.BinOp):
        return True
    if isinstance(parent, ast.Compare):
        return True
    if isinstance(parent, ast.UnaryOp):
        return True
    return False


_PRESENTATION_CALLS = {
    "figsize",
    "legend",
    "tick_params",
    "set_title",
    "set_xlabel",
    "set_ylabel",
    "set_xlim",
    "set_ylim",
    "savefig",
    "tight_layout",
    "colorbar",
    "axis",
    "text",
    "subplots",
}


def _is_presentation_context(tree: ast.AST, node: ast.Constant) -> bool:
    parent = _parent_of(tree, node)
    while parent is not None:
        if isinstance(parent, ast.Call):
            name = getattr(parent.func, "id", "") or getattr(parent.func, "attr", "")
            return name in _PRESENTATION_CALLS
        if isinstance(parent, (ast.keyword, ast.Tuple, ast.List, ast.Set)):
            parent = _parent_of(tree, parent)
            continue
        return False
    return False


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
        "Configured scientific values duplicated in production source:\n" + "\n".join(violations)
    )


def test_experiment_axes_not_duplicated_in_python() -> None:
    raw = yaml.safe_load((ROOT / "config" / "experiments.yaml").read_text(encoding="utf-8"))
    axis_values: set[object] = set()
    for experiment in raw["experiments"]:
        for _axis, values in (experiment.get("axes") or {}).items():
            if isinstance(values, list):
                axis_values.update(values)
        for cell in experiment.get("coupled_cells") or []:
            axis_values.update(cell.values())
    axis_values.discard(0)
    axis_values.discard(0.0)
    axis_values.discard(1)
    axis_values.discard(1.0)
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                continue
            if float(node.value) in {
                float(item)
                for item in axis_values
                if isinstance(item, (int, float)) and not isinstance(item, bool)
            }:
                parent = _parent_of(tree, node)
                if _is_structural_context(parent, node):
                    continue
                if (relative, float(node.value)) in _ALLOWED_SOURCE_LITERALS:
                    continue
                raise AssertionError(
                    f"{relative}:{getattr(node, 'lineno', 0)} duplicates experiment axis "
                    f"value {node.value!r}"
                )


def _is_seed_position(parent: ast.AST | None, node: ast.Constant) -> bool:
    if parent is None:
        return False
    if isinstance(parent, ast.keyword) and parent.arg is not None and "seed" in parent.arg:
        return True
    if isinstance(parent, ast.Call):
        name = getattr(parent.func, "id", "") or getattr(parent.func, "attr", "")
        if "seed" in name:
            return True
    if isinstance(parent, (ast.Tuple, ast.List, ast.Set)):
        elements = parent.elts
        if elements and all(
            isinstance(item, ast.Constant)
            and isinstance(item.value, int)
            and not isinstance(item.value, bool)
            for item in elements
        ):
            return True
    return False


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
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, int)
                and not isinstance(node.value, bool)
            ):
                if node.value in seeds:
                    parent = _parent_of(tree, node)
                    if _is_structural_context(parent, node):
                        continue
                    if not _is_seed_position(parent, node):
                        continue
                    raise AssertionError(
                        f"{relative}:{getattr(node, 'lineno', 0)} duplicates configured "
                        f"seed {node.value!r}"
                    )
