from typing import Union, Type, Dict

from fisheye.configs import (
    BaseModelConfig,
    YOLOv5ModelConfig,
    YOLOv11ModelConfig,
    YOLOv26ModelConfig,
)
from fisheye.configs.models import (
    BaseLengthModelConfig,
    UNetLengthModelConfig,
    HeatmapCNNLengthModelConfig,
)
from fisheye.enums import DetectorType, LengthEstimatorType


DETECTOR_MODEL_CONFIG_REGISTRY: Dict[DetectorType, Type[BaseModelConfig]] = {
    DetectorType.YOLOv5: YOLOv5ModelConfig,
    DetectorType.YOLOv11: YOLOv11ModelConfig,
    DetectorType.YOLOv26: YOLOv26ModelConfig,
}

LENGTH_MODEL_CONFIG_REGISTRY: Dict[LengthEstimatorType, Type[BaseLengthModelConfig]] = {
    LengthEstimatorType.UNET: UNetLengthModelConfig,
    LengthEstimatorType.HEATMAP_CNN: HeatmapCNNLengthModelConfig,
}


def get_detector_config(
    model_type: Union[DetectorType, str], **kwargs
) -> BaseModelConfig:
    """Return the appropriate config class for the given detector type."""
    if not isinstance(model_type, DetectorType):
        model_type = DetectorType(model_type)

    config_cls = DETECTOR_MODEL_CONFIG_REGISTRY[model_type]

    return config_cls(**kwargs)


def get_length_model_config(
    model_type: Union[LengthEstimatorType, str], **kwargs
) -> BaseLengthModelConfig:
    """Return the appropriate config class for the given length model type."""
    if not isinstance(model_type, LengthEstimatorType):
        model_type = LengthEstimatorType(model_type)

    config_cls = LENGTH_MODEL_CONFIG_REGISTRY[model_type]

    return config_cls(**kwargs)
