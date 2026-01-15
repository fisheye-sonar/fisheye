from pathlib import Path
from typing import Dict, Type, Any, Optional

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
) -> Optional[BaseLengthEstimator]:
    """
    Factory function to create a length estimator based on configuration.

    Args:
        config: Length model configuration (BaseLengthModelConfig or subclass)
        metadata: Dataset metadata (ARISMetadata)

    Returns:
        An instance of a length estimator, or None if configuration is invalid/weights missing.

    Raises:
        ValueError: If model type is not recognized but valid checks pass (shouldn't happen with current logic)
    """
    model_type = config.type
    weights = config.weights

    if not weights:
        logger.warning(
            "no_weights_provided",
            message="Length estimator will not be used. Default lengths to 0.",
        )
        return None

    weights = Path(weights).resolve()
    if not weights.exists():
        logger.warning(
            "weights_path_not_found",
            message="Length estimator will not be used. Default lengths to 0.",
            path=str(weights),
        )
        return None

    try:
        model_type_enum = LengthEstimatorType(model_type)

    except ValueError:
        valid_types = [e.value for e in LengthEstimatorType]
        logger.warning(
            "invalid_model_type",
            message="Length estimator will not be used. Default lengths to 0.",
            model_type=model_type,
            valid_types=valid_types,
        )
        return None

    estimator_cls = LENGTH_ESTIMATOR_REGISTRY.get(model_type_enum)

    if estimator_cls is None:
        logger.warning(
            "model_type_not_implemented",
            message="Length estimator will not be used. Default lengths to 0.",
            model_type=model_type,
        )
        return None

    logger.info(
        "initialized_length_estimator",
        model_type=model_type_enum.value,
        estimator_class=estimator_cls.__name__,
    )

    try:
        return estimator_cls(metadata, config)

    except Exception as e:
        logger.error(
            "failed_to_initialize_estimator",
            error=str(e),
            model_type=model_type_enum.value,
        )
        return None
