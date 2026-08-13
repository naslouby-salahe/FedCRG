"""Rich console rendering for long-running campaign and experiment execution."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_console = Console()


def render_campaign_status(
    campaign_id: str,
    experiments: tuple[tuple[str, str], ...],
    current_experiment: str | None,
    current_stage: str | None,
    elapsed_seconds: float,
) -> None:
    """Render one campaign status panel: experiment id + status per row."""
    table = Table(title=f"campaign {campaign_id}", show_header=True)
    table.add_column("experiment")
    table.add_column("status")
    for experiment_id, status in experiments:
        table.add_row(experiment_id, status)
    panel = Panel(
        table,
        title=f"stage: {current_stage or 'starting'}",
        subtitle=f"current: {current_experiment or '-'} | elapsed {elapsed_seconds:.0f}s",
    )
    _console.print(panel)


def render_cache_status(
    *,
    preprocessing_hit: bool,
    model_hit: bool,
    score_hit: bool,
) -> None:
    """Render one compact cache reuse line: preprocessing/model/score hit-or-miss."""
    hits = {
        "preprocessing": "hit" if preprocessing_hit else "miss",
        "model": "hit" if model_hit else "miss",
        "score": "hit" if score_hit else "miss",
    }
    line = " | ".join(f"{key}={value}" for key, value in hits.items())
    _console.print(f"[dim]{line}[/dim]")
