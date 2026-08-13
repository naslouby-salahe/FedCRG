"""Resolve composable YAML documents into one immutable typed experiment config."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pydantic import ValidationError

from fedcrg.config.dataset_config import DatasetConfig
from fedcrg.config.detector_config import DetectorConfig
from fedcrg.config.experiment_config import ExperimentConfig
from fedcrg.config.load import load_yaml
from fedcrg.config.method_config import ProtocolConfig
from fedcrg.config.statistics_config import StatisticsConfig
from fedcrg.config.training_config import RandomnessConfig, TrainingConfig
from fedcrg.config.validate import validate_experiment_config
from fedcrg.domain.enums import DetectorId, ExperimentId
from fedcrg.domain.errors import ConfigurationError

_SECTION_KEYS = (
    "protocol",
    "dataset",
    "detector",
    "training",
    "randomness",
    "statistics",
)


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
            detector: DetectorConfig
            if detector_id is DetectorId.AUTOENCODER:
                from fedcrg.config.detector_config import AutoencoderConfig

                detector = AutoencoderConfig.model_validate(detector_raw)
            elif detector_id is DetectorId.DEEP_SVDD:
                from fedcrg.config.detector_config import DeepSvddConfig

                detector = DeepSvddConfig.model_validate(detector_raw)
            else:
                raise ConfigurationError(f"Unsupported detector id: {detector_id.value}")
            policies_raw = root["policies"]
            if not isinstance(policies_raw, list):
                raise ConfigurationError("policies must be a YAML list")
            outputs_raw = root.get("outputs_root", "outputs")
            if not isinstance(outputs_raw, str):
                raise ConfigurationError("outputs_root must be a path string")
            preprocessed_raw = root.get("preprocessed_root", "data/preprocessed")
            if not isinstance(preprocessed_raw, str):
                raise ConfigurationError("preprocessed_root must be a path string")
            return ExperimentConfig(
                id=ExperimentId(str(root["id"])),
                protocol=protocol,
                dataset=dataset,
                detector=detector,
                training=TrainingConfig.model_validate(
                    _resolve_section(root["training"], "training")
                ),
                randomness=RandomnessConfig.model_validate(
                    _resolve_section(root["randomness"], "randomness")
                ),
                statistics=StatisticsConfig.model_validate(
                    _resolve_section(root["statistics"], "statistics")
                ),
                policies=tuple(policies_raw),
                outputs_root=Path(outputs_raw),
                preprocessed_root=Path(preprocessed_raw),
            )
        except (KeyError, ValueError, ValidationError, ConfigurationError) as exc:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError(f"Invalid experiment configuration {path}: {exc}") from exc


def load_config(path: Path | str) -> ExperimentConfig:
    config = ExperimentConfigResolver().resolve(path)
    validate_experiment_config(config)
    return config
