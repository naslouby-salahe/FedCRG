"""Output-layout and prepared-column contract.

No production module may hardcode an outputs/ path fragment or a prepared-data
column name: every reserved path is owned by ``OutputsLayout`` in
``evidence/store.py``, and every prepared column is a member of the
``PreparedColumn`` enum in ``types.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "fedcrg"

_PATH_OWNER = "evidence/store.py"
# The owner module may name the literal fragments once; everyone else must use
# the layout properties.
_FORBIDDEN_FRAGMENTS = (
    '"runs"',
    '"cache"',
    '"models"',
    '"scores"',
    '"analysis"',
    '"campaigns"',
    '"logs"',
    '"monitoring"',
    '"telemetry.jsonl"',
    '"benchmark.json"',
    '"readiness_plans.json"',
    '"mismatch_cutoffs.json"',
    '"environment.json"',
)
_COLUMN_OWNER = "types.py"


def _python_files() -> tuple[Path, ...]:
    return tuple(sorted(SRC.rglob("*.py")))


def test_no_module_hardcodes_an_outputs_path_fragment() -> None:
    violations: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        if relative == _PATH_OWNER:
            continue
        source = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            for line_number, line in enumerate(source.splitlines(), 1):
                if fragment not in line:
                    continue
                stripped = line.strip()
                if stripped.startswith("@click.group"):
                    continue
                if "Path(" not in line and ' / "' not in line and '" / ' not in line:
                    continue
                violations.append(f"{relative}:{line_number}: {fragment}")
    assert not violations, (
        "Outputs path fragments must be owned by OutputsLayout "
        "in evidence/store.py, not hardcoded:\n" + "\n".join(violations)
    )


def test_prepared_column_names_come_from_the_enum() -> None:
    expected = {
        "row_id",
        "role",
        "label",
        "attack_group",
        "source_file",
        "source_row_index",
        "capture_time",
    }
    for path in _python_files():
        relative = path.relative_to(SRC).as_posix()
        if relative == _COLUMN_OWNER:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in expected:
                    violations = f"{relative}:{node.lineno}: literal {node.value!r}"
                    raise AssertionError(
                        "Prepared column names must come from the PreparedColumn enum:\n"
                        + violations
                    )


def test_outputs_layout_owns_every_reserved_path() -> None:
    from fedcrg.evidence.store import OutputsLayout

    layout = OutputsLayout(Path("outputs"))
    assert layout.runs == Path("outputs/runs")
    assert layout.cache_models == Path("outputs/cache/models")
    assert layout.cache_scores == Path("outputs/cache/scores")
    assert layout.cache_analysis == Path("outputs/cache/analysis")
    assert layout.campaigns == Path("outputs/campaigns")
    assert layout.logs == Path("outputs/logs")
    assert layout.monitoring == Path("outputs/monitoring")
