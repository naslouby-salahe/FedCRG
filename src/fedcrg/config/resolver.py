"""Resolve composable experiment configuration into one immutable model."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fedcrg.config.loader import load_yaml
from fedcrg.config.models import (
    AutoencoderConfig,
    DatasetConfig,
    DeepSvddConfig,
    ExperimentConfig,
    ProtocolConfig,
    RandomnessConfig,
    TrainingConfig,
)
from fedcrg.core.enums import DetectorId
from fedcrg.core.exceptions import ConfigurationError

_SECTION_KEYS = ("protocol", "dataset", "detector")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_experiment_document(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        cycle = " -> ".join(str(item) for item in (*stack, resolved))
        raise ConfigurationError(f"Configuration inheritance cycle: {cycle}")
    root = load_yaml(resolved)
    for key in _SECTION_KEYS:
        value = root.get(key)
        if isinstance(value, str):
            root[key] = str((resolved.parent / value).resolve())
    parent = root.pop("extends", None)
    if parent is None:
        return root
    if not isinstance(parent, str):
        raise ConfigurationError("extends must be a relative YAML path")
    parent_document = _load_experiment_document(resolved.parent / parent, (*stack, resolved))
    return _deep_merge(parent_document, root)


def _resolve_section(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return load_yaml(Path(value))
    if isinstance(value, dict):
        return value
    raise ConfigurationError("Configuration section must be a mapping or YAML path")


class ExperimentConfigResolver:
    def resolve(self, path: Path | str) -> ExperimentConfig:
        root = _load_experiment_document(Path(path))
        try:
            protocol = ProtocolConfig.model_validate(_resolve_section(root["protocol"]))
            dataset = DatasetConfig.model_validate(_resolve_section(root["dataset"]))
            detector_raw = _resolve_section(root["detector"])
            detector_id = DetectorId(detector_raw["id"])
            detector = (
                AutoencoderConfig.model_validate(detector_raw)
                if detector_id is DetectorId.AUTOENCODER
                else DeepSvddConfig.model_validate(detector_raw)
            )
            return ExperimentConfig(
                id=root["id"],
                protocol=protocol,
                dataset=dataset,
                detector=detector,
                training=TrainingConfig.model_validate(root["training"]),
                randomness=RandomnessConfig.model_validate(root.get("randomness", {})),
                policies=tuple(root["policies"]),
                outputs_root=Path(root.get("outputs_root", "outputs")),
            )
        except (KeyError, ValueError, ValidationError) as exc:
            raise ConfigurationError(f"Invalid experiment configuration {path}: {exc}") from exc
