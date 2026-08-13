from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fedcrg"


def _python_files() -> tuple[Path, ...]:
    return tuple(sorted(SRC.rglob("*.py")))


def test_legacy_repository_layers_do_not_return() -> None:
    assert not (ROOT / "scripts").exists()
    assert not (ROOT / "fedcrg").exists()
    assert not (ROOT / "setup.py").exists()


def test_source_module_names_describe_capabilities_not_paper_shorthand() -> None:
    forbidden = {"gate_a.py", "gate_b.py", "baselines", "fl"}
    for path in SRC.rglob("*"):
        assert path.name.lower() not in forbidden


def test_production_source_has_no_personal_absolute_paths() -> None:
    forbidden_fragments = (
        "/home/naslouby",
        "d:\\projects",
        "c:\\users",
    )
    for path in _python_files():
        text = path.read_text(encoding="utf-8").lower()
        for fragment in forbidden_fragments:
            assert fragment not in text, f"personal path leaked into {path}"


def test_production_source_has_no_semicolon_compressed_statements() -> None:
    for path in _python_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert ";" not in stripped, f"compressed multi-statement line in {path}:{line_number}"
