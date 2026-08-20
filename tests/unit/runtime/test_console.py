"""Tests for console status rendering helpers."""

from __future__ import annotations

import pytest

from fedcrg.runtime import render_cache_status


def test_render_cache_status_prints_outcome_and_target(capsys: pytest.CaptureFixture[str]) -> None:
    """Cache status output distinguishes hits from misses and names the target."""
    render_cache_status(cache_kind="preprocessed", hit=True, target="nbaiot")
    render_cache_status(cache_kind="model", hit=False, target="nb01", detail="rebuild")
    output = capsys.readouterr().out
    assert "[hit]" in output
    assert "[miss]" in output
    assert "preprocessed nbaiot" in output
    assert "model nb01 (rebuild)" in output
