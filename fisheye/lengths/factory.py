from typing import Any
from fisheye.configs.inference import LengthModelConfig
from fisheye.lengths.base import BaseLengthEstimator
from fisheye.lengths.estimator import UNetLengthEstimator


def create_length_estimator(
    config: LengthModelConfig, metadata: Any
) -> BaseLengthEstimator:
    """
    Factory function to create a length estimator based on configuration.

    Args:
        config: Length configuration containing model config.
        metadata: Dataset metadata.

    Returns:
        An instance of a length estimator.
    """
    model_type = config.model_config.model_type

    if model_type == "unet":
        return UNetLengthEstimator(metadata, config.model_config)
    else:
        raise ValueError(f"Unknown length estimator model type: {model_type}")
