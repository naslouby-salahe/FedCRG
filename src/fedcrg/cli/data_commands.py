"""Data-preparation and environment CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from fedcrg.artifacts.environment import EnvironmentLocker
from fedcrg.configuration.resolve import load_config
from fedcrg.experiments.dataset_preparation import (
    PrepareData,
    PrepareDiadFeatureSensitivity,
)

_DATASET_CONFIG_PATHS = {
    "nbaiot": "configs/experiments/primary/nbaiot.yaml",
    "diad": "configs/experiments/external/diad.yaml",
}


def _config_for_dataset(dataset_id: str) -> Path:
    relative = _DATASET_CONFIG_PATHS.get(dataset_id)
    if relative is None:
        raise click.BadParameter(
            f"Unknown dataset id {dataset_id!r}. Expected one of "
            + ", ".join(sorted(_DATASET_CONFIG_PATHS))
        )
    return Path(relative)


@click.group(name="data")
def data_group() -> None:
    """Prepare and freeze seed-independent dataset caches and role-assignment manifests."""


@data_group.command(name="preprocess")
@click.argument("dataset_id", required=False)
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--data-root", type=click.Path(path_type=Path, exists=True), required=True)
def preprocess_data(dataset_id: str | None, config_path: Path | None, data_root: Path) -> None:
    """Preprocess [DATASET_ID] into data/preprocessed/ (or --config for a custom profile)."""
    if dataset_id is not None:
        resolved = _config_for_dataset(dataset_id)
    elif config_path is not None:
        resolved = config_path
    else:
        raise click.BadParameter("Provide DATASET_ID or --config")
    cache_root = PrepareData().prepare(load_config(resolved), data_root)
    click.echo(str(cache_root))


@data_group.command(name="status")
@click.argument("dataset_id", required=False)
def data_status(dataset_id: str | None) -> None:
    """Show prepared-cache status for one dataset (or all datasets)."""
    import pandas as pd

    root = Path("data/preprocessed")
    if not root.exists():
        click.echo(json.dumps({"prepared": []}, indent=2))
        return
    rows: list[dict[str, bool | str]] = []
    for dataset_root in sorted(path for path in root.iterdir() if path.is_dir()):
        if dataset_id is not None and dataset_root.name != dataset_id:
            continue
        for identity in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            manifest = identity / "manifest.json"
            rows.append(
                {
                    "dataset": dataset_root.name,
                    "identity": identity.name,
                    "prepared": manifest.is_file(),
                }
            )
    if not rows:
        click.echo(json.dumps({"prepared": []}, indent=2))
        return
    frame = pd.DataFrame.from_records(rows)
    click.echo(frame.to_string(index=False))


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
    help="Frozen diad_eligibility.json from a prior `data preprocess` run.",
)
@click.option("--feature-manifest", type=click.Path(path_type=Path), required=True)
def prepare_feature_sensitivity(
    config_path: Path,
    data_root: Path,
    eligibility_manifest: Path,
    feature_manifest: Path,
) -> None:
    """Freeze the R14 training-schema-only DIAD feature contract and prepare its cache."""
    config, cache_root = PrepareDiadFeatureSensitivity().prepare(
        load_config(config_path),
        data_root,
        eligibility_manifest,
        feature_manifest,
    )
    click.echo(f"config_hash={config.config_hash}\ncache={cache_root}")


@click.group(name="environment")
def environment_group() -> None:
    """Freeze the validated Python environment before confirmatory runs."""


@environment_group.command(name="freeze")
@click.option("--output", type=click.Path(path_type=Path), default=Path("requirements.lock"))
def freeze_environment(output: Path) -> None:
    lock = EnvironmentLocker().freeze(output)
    click.echo(f"lock={lock.path}\nsha256={lock.sha256.value}")
