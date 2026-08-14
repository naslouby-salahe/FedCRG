"""Unit tests for the console renderers used during long runs."""

from __future__ import annotations

from fedcrg.runtime import render_cache_status, render_campaign_status


def test_render_campaign_status_accepts_progress_counts() -> None:
    render_campaign_status(
        campaign_id="c1",
        status="running",
        completed=1,
        total=2,
        current_experiment="external_diad",
        elapsed_seconds=12.0,
    )


def test_render_cache_status_accepts_hit_flags() -> None:
    render_cache_status(cache_kind="preprocessed", hit=True, target="nbaiot")
    render_cache_status(cache_kind="model", hit=False, target="nb01", detail="rebuild")
