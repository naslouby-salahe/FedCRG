from __future__ import annotations

import pytest

from fedcrg.runtime import render_cache_status, render_campaign_status


def test_render_campaign_status_prints_progress_row(capsys: pytest.CaptureFixture[str]) -> None:
    render_campaign_status(
        campaign_id="c1",
        status="running",
        completed=1,
        total=2,
        current_experiment="external_diad",
        elapsed_seconds=12.0,
    )
    output = capsys.readouterr().out
    assert "c1" in output
    assert "running" in output
    assert "external_diad" in output


def test_render_cache_status_prints_outcome_and_target(capsys: pytest.CaptureFixture[str]) -> None:
    render_cache_status(cache_kind="preprocessed", hit=True, target="nbaiot")
    render_cache_status(cache_kind="model", hit=False, target="nb01", detail="rebuild")
    output = capsys.readouterr().out
    assert "[hit]" in output
    assert "[miss]" in output
    assert "preprocessed nbaiot" in output
    assert "model nb01 (rebuild)" in output
