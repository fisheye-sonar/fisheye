from typing import Dict, Type, Any

import structlog

from fisheye.configs.models import BaseLengthModelConfig
from fisheye.enums import LengthEstimatorType
from fisheye.lengths.base import BaseLengthEstimator
from fisheye.lengths.estimator import UNetLengthEstimator


logger = structlog.getLogger(__name__)


LENGTH_ESTIMATOR_REGISTRY: Dict[LengthEstimatorType, Type[BaseLengthEstimator]] = {
    LengthEstimatorType.UNET: UNetLengthEstimator,
}


def create_length_estimator(
    config: BaseLengthModelConfig, metadata: Any
) -> BaseLengthEstimator:
    """
    Factory function to create a length estimator based on configuration.

    Args:
        config: Length model configuration (BaseLengthModelConfig or subclass)
        metadata: Dataset metadata (ARISMetadata)

    Returns:
        An instance of a length estimator

    Raises:
        ValueError: If model type is not recognized or not in registry
    """
    model_type_str = config.type

    try:
        model_type = LengthEstimatorType(model_type_str)
    except ValueError:
        raise ValueError(
            f"Unknown length estimator model type: '{model_type_str}'. "
            f"Must be one of: {[e.value for e in LengthEstimatorType]}"
        )

    estimator_cls = LENGTH_ESTIMATOR_REGISTRY.get(model_type)

    if estimator_cls is None:
        raise ValueError(
            f"Length estimator '{model_type.value}' not implemented. "
            f"Available: {list(LENGTH_ESTIMATOR_REGISTRY.keys())}"
        )

    logger.info(
        "initialized_length_estimator",
        model_type=model_type.value,
        estimator_class=estimator_cls.__name__,
    )

    return estimator_cls(metadata, config)
