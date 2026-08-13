"""Unit tests for the Rich console renderers used during long runs."""

from __future__ import annotations

from fedcrg.domain.enums import CampaignStatusValue, ExperimentId
from fedcrg.runtime.console import render_cache_status, render_campaign_status


def test_render_campaign_status_accepts_experiment_rows() -> None:
    render_campaign_status(
        "c1",
        (
            (ExperimentId.PRIMARY_NBAIOT, CampaignStatusValue.COMPLETE),
            (ExperimentId.EXTERNAL_DIAD, CampaignStatusValue.RUNNING),
        ),
        current_experiment=ExperimentId.EXTERNAL_DIAD,
        current_stage="running external_diad",
        elapsed_seconds=12.0,
    )


def test_render_cache_status_accepts_hit_flags() -> None:
    render_cache_status(preprocessing_hit=True, model_hit=False, score_hit=True)
