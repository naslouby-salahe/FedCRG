from pathlib import Path


def test_production_paths_use_domain_responsibility_names() -> None:
    root = Path("src/fedcrg")
    forbidden = {"gate_a.py", "gate_b.py", "states.py", "fedcrg.py"}
    paths = {path.name for path in root.rglob("*.py")}
    assert not forbidden.intersection(paths)
