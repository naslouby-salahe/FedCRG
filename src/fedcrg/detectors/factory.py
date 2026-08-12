"""Detector construction."""

from fedcrg.config.models import AutoencoderConfig, DeepSvddConfig, DetectorConfig
from fedcrg.detectors.autoencoder import Autoencoder
from fedcrg.detectors.base import DetectorModel
from fedcrg.detectors.deep_svdd import DeepSvdd


class DetectorFactory:
    def create(self, input_dim: int, config: DetectorConfig) -> DetectorModel:
        if isinstance(config, AutoencoderConfig):
            return Autoencoder(input_dim, config)
        if isinstance(config, DeepSvddConfig):
            return DeepSvdd(input_dim, config)
        raise TypeError(f"Unsupported detector configuration: {type(config)!r}")
