"""Contract tests for the target repository architecture.

These tests define the target tree and forbid the obsolete architecture from
returning. They are the enforcement contract for the restructure.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fedcrg"


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def test_only_intended_production_packages_exist() -> None:
    allowed_dirs = {
        "data",
        "learning",
        "thresholding",
        "experiments",
        "evidence",
    }
    for path in sorted(SRC.iterdir()):
        if path.is_dir() and path.name not in allowed_dirs:
            if path.name == "__pycache__":
                continue
            raise AssertionError(f"Unexpected production package: {path.name}")


def test_expected_top_level_modules_exist() -> None:
    for name in (
        "__init__.py",
        "__main__.py",
        "cli.py",
        "types.py",
        "config.py",
        "reporting.py",
        "runtime.py",
    ):
        assert (SRC / name).is_file(), f"Missing top-level module {name}"


def test_expected_package_modules_exist() -> None:
    expected = {
        "data": ("__init__.py", "datasets.py", "preprocessing.py"),
        "learning": ("__init__.py", "detectors.py", "federated.py", "scores.py"),
        "thresholding": ("__init__.py", "readiness.py", "policies.py", "metrics.py"),
        "experiments": ("__init__.py", "runner.py", "analyses.py"),
        "evidence": ("__init__.py", "models.py", "store.py"),
    }
    for package, files in expected.items():
        for name in files:
            assert (SRC / package / name).is_file(), f"Missing {package}/{name}"


def test_obsolete_packages_are_absent() -> None:
    obsolete = {
        "domain",
        "configuration",
        "artifacts",
        "datasets",
        "decision",
        "detectors",
        "evaluation",
        "federation",
        "scoring",
        "analysis",
        "reporting",
        "runtime",
        "cli",
    }
    for name in obsolete:
        assert not (SRC / name).exists(), f"Obsolete package {name} still exists"
    assert not (SRC / "decision" / "policies").exists()
    assert not (SRC / "experiments" / "definitions").exists()


def test_obsolete_module_files_are_absent() -> None:
    forbidden = {
        "experiment_definition.py",
        "gate_a.py",
        "gate_b.py",
        "utils.py",
        "helpers.py",
        "common.py",
    }
    found = {path.name for path in _python_files(SRC)} & forbidden
    assert not found, f"Forbidden module files present: {sorted(found)}"


def test_config_root_has_exactly_three_documents() -> None:
    config_root = ROOT / "config"
    assert config_root.is_dir(), "config/ root must exist"
    names = sorted(path.name for path in config_root.iterdir() if path.is_file())
    assert names == ["datasets.yaml", "experiments.yaml", "study.yaml"], names


def test_legacy_configs_tree_is_absent() -> None:
    assert not (ROOT / "configs").exists(), "configs/ hierarchy must be removed"


def test_preprocessed_and_raw_data_roots() -> None:
    assert (ROOT / "data" / "preprocessed").is_dir()
    assert (ROOT / "data" / "raw").is_dir()


def test_output_roots_conform_to_target() -> None:
    outputs = ROOT / "outputs"
    for relative in (
        "logs",
        "monitoring",
        "cache/models",
        "cache/scores",
        "cache/analysis",
        "runs",
        "campaigns",
        "figures",
        "reports",
    ):
        assert (outputs / relative).is_dir(), f"Missing outputs/{relative}"
    for stale in ("cache/datasets", "cache/precomputed", "reports/latest", "reports/publication"):
        assert not (outputs / stale).exists(), f"Stale output root outputs/{stale}"


def test_publication_results_root() -> None:
    assert (ROOT / "results").is_dir(), "results/ root must exist"


def test_no_redirect_modules() -> None:
    import ast

    for path in _python_files(SRC):
        if path.name in ("__init__.py", "__main__.py"):
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
                f"{path.relative_to(ROOT)} is a redirect module of {source_import.module}"
            )


def test_no_vague_module_names() -> None:
    import ast

    vague = {
        "utils",
        "helpers",
        "common",
        "manager",
        "handler",
        "processor",
        "engine",
        "service",
        "base",
        "registry",
        "factory",
    }
    for path in _python_files(SRC):
        if path.name in ("__init__.py", "__main__.py"):
            continue
        stem = path.stem
        assert stem not in vague, f"Vague module name {stem}"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                lowered = node.name.lower()
                for token in vague:
                    assert not lowered.endswith(token), (
                        f"{path.relative_to(ROOT)} defines {node.name}, a vague {token} class name"
                    )


def test_no_unnecessary_subpackages() -> None:
    for path in sorted(SRC.iterdir()):
        if path.is_dir():
            members = [p for p in path.iterdir() if p.name != "__init__.py" and p.suffix == ".py"]
            if len(members) == 1:
                raise AssertionError(
                    f"Package {path.name} contains a single trivial module {members[0].name}"
                )
