from __future__ import annotations

import io
import tokenize
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
        source = path.read_text(encoding="utf-8")
        # Collect line numbers that belong to string literals (docstrings and
        # prose), where a semicolon is text, not a statement separator.
        string_lines: set[int] = set()
        try:
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            for token in tokens:
                if token.type == tokenize.STRING:
                    string_lines.update(range(token.start[0], token.end[0] + 1))
        except tokenize.TokenError:
            pass
        for line_number, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if line_number in string_lines:
                continue
            assert ";" not in stripped, f"compressed multi-statement line in {path}:{line_number}"
