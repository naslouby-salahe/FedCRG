"""Dataset preparation CLI commands."""

from __future__ import annotations

from pathlib import Path

import click

from fedcrg.pipeline.prepare_dataset import PrepareData
from fedcrg.config.resolve import load_config


@click.group(name="data")
def data_group() -> None:
    """Prepare and freeze seed-independent dataset caches and role-assignment manifests."""


@data_group.command(name="prepare")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--data-root", type=click.Path(path_type=Path, exists=True), required=True)
def prepare_data(config_path: Path, data_root: Path) -> None:
    cache_root = PrepareData().prepare(load_config(config_path), data_root)
    click.echo(str(cache_root))


@data_group.command(name="prepare-feature-sensitivity")
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--data-root", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--eligibility-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Frozen diad_eligibility.json from a prior `data prepare` run.",
)
@click.option("--feature-manifest", type=click.Path(path_type=Path), required=True)
def prepare_feature_sensitivity(
    config_path: Path,
    data_root: Path,
    eligibility_manifest: Path,
    feature_manifest: Path,
) -> None:
    """Freeze the R14 training-schema-only DIAD feature contract and prepare its cache."""
    from fedcrg.pipeline.prepare_dataset import PrepareDiadFeatureSensitivity

    config, cache_root = PrepareDiadFeatureSensitivity().prepare(
        load_config(config_path),
        data_root,
        eligibility_manifest,
        feature_manifest,
    )
    click.echo(f"config_hash={config.config_hash}\ncache={cache_root}")
