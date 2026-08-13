"""Resolve composable YAML documents into one immutable typed experiment config."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

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


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{context} must be a string-keyed mapping")
    return {str(key): item for key, item in value.items()}


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    result = deepcopy(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(_mapping(current, key), _mapping(value, key))
        else:
            result[key] = deepcopy(value)
    return result


def _load_experiment_document(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, object]:
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


def _resolve_section(value: object, context: str) -> dict[str, object]:
    if isinstance(value, str):
        return load_yaml(Path(value))
    return _mapping(value, context)


class ExperimentConfigResolver:
    def resolve(self, path: Path | str) -> ExperimentConfig:
        root = _load_experiment_document(Path(path))
        try:
            protocol = ProtocolConfig.model_validate(_resolve_section(root["protocol"], "protocol"))
            dataset = DatasetConfig.model_validate(_resolve_section(root["dataset"], "dataset"))
            detector_raw = _resolve_section(root["detector"], "detector")
            detector_id = DetectorId(str(detector_raw["id"]))
            detector = (
                AutoencoderConfig.model_validate(detector_raw)
                if detector_id is DetectorId.AUTOENCODER
                else DeepSvddConfig.model_validate(detector_raw)
            )
            policies_raw = root["policies"]
            if not isinstance(policies_raw, list):
                raise ConfigurationError("policies must be a YAML list")
            outputs_raw = root.get("outputs_root", "outputs")
            if not isinstance(outputs_raw, str):
                raise ConfigurationError("outputs_root must be a path string")
            return ExperimentConfig(
                id=root["id"],
                protocol=protocol,
                dataset=dataset,
                detector=detector,
                training=TrainingConfig.model_validate(_mapping(root["training"], "training")),
                randomness=RandomnessConfig.model_validate(
                    _mapping(root.get("randomness", {}), "randomness")
                ),
                policies=tuple(policies_raw),
                outputs_root=Path(outputs_raw),
            )
        except (KeyError, ValueError, ValidationError, ConfigurationError) as exc:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError(f"Invalid experiment configuration {path}: {exc}") from exc
