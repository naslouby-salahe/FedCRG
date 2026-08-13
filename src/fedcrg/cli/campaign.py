"""Campaign execution and status CLI."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.experiments.campaign import CampaignRunner, CampaignStatusStore, CampaignWorkItem


@click.group(name="campaign")
def campaign_group() -> None:
    """Run and inspect persistent research campaigns."""


@campaign_group.command(name="run")
@click.option("--campaign-id", required=True)
@click.option(
    "--config", "config_paths", type=click.Path(path_type=Path, exists=True), multiple=True
)
@click.option("--prepared-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
@click.option("--results", "results_root", type=click.Path(path_type=Path), default=Path("results"))
def campaign_run(
    campaign_id: str,
    config_paths: tuple[Path, ...],
    prepared_root: Path,
    outputs: Path,
    results_root: Path,
) -> None:
    """Execute a campaign over the given experiment configs and record persistent status."""
    if not config_paths:
        raise click.UsageError("At least one --config is required")
    from fedcrg.config.resolve import load_config

    work_items = tuple(
        CampaignWorkItem(
            experiment_id=load_config(path).id,
            config_path=path,
            prepared_root=prepared_root,
        )
        for path in config_paths
    )
    status = CampaignRunner().run(
        campaign_id,
        work_items,
        outputs_root=outputs,
        results_root=results_root,
    )
    click.echo(json.dumps(status.to_dict(), indent=2))


@campaign_group.command(name="status")
@click.option("--campaign-id", required=True)
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
def campaign_status(campaign_id: str, outputs: Path) -> None:
    """Show persistent status for one campaign."""
    store = CampaignStatusStore(outputs / "campaigns")
    status = store.load(campaign_id)
    click.echo(json.dumps(status.to_dict(), indent=2))


@campaign_group.command(name="list")
@click.option("--outputs", type=click.Path(path_type=Path), default=Path("outputs"))
def campaign_list(outputs: Path) -> None:
    """List recorded campaigns."""
    campaigns_root = outputs / "campaigns"
    if not campaigns_root.exists():
        click.echo("[]")
        return
    ids = sorted(path.stem for path in campaigns_root.glob("*.json"))
    click.echo(json.dumps(ids, indent=2))
