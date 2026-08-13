"""Detector construction from configuration."""

from fedcrg.config.training_config import AutoencoderConfig, DeepSvddConfig, DetectorConfig
from fedcrg.detectors.autoencoder import Autoencoder
from fedcrg.detectors.deep_svdd import DeepSvdd
from fedcrg.detectors.detector import DetectorModel


def create_detector(input_dim: int, config: DetectorConfig) -> DetectorModel:
    if isinstance(config, AutoencoderConfig):
        return Autoencoder(input_dim, config)
    if isinstance(config, DeepSvddConfig):
        return DeepSvdd(input_dim, config)
    raise TypeError(f"Unsupported detector configuration: {type(config)!r}")
