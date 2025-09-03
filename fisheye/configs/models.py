from dataclasses import dataclass
from typing import Optional, Union, List, Type, Dict

from fisheye.detect.base import BaseModel
from fisheye.enums import DeviceType, DetectorType


@dataclass
class BaseModelConfig:
    """Base model configuration."""

    type: Optional[str] = None
    weights: Union[str, BaseModel] = None
    device: str = DeviceType.MPS.value


@dataclass
class YOLOv5ModelConfig(BaseModelConfig):
    """YOLOv5 model config.

    Exposing class variables from YOLOv5's AutoShape.
    """

    type: str = DetectorType.YOLOv5.value
    agnostic: bool = False  # NMS class-agnostic
    multi_label: bool = False  # NMS multiple labels per box
    classes: Optional[List[int]] = (
        None  # (Optional list) filter by class, i.e. = [0, 15, 16] for COCO
    )
    max_det: int = 300  # Maximum number of detections per image
    amp: bool = False  # Automatic Mixed Precision (AMP) inference


@dataclass
class YOLOv11ModelConfig(BaseModelConfig):
    """YOLOv11 model config.

    Exposing class variables from YOLOv5's AutoShape.
    """

    type: str = DetectorType.YOLOv11.value


DETECTOR_CONFIG_REGISTRY: Dict[DetectorType, Type[BaseModelConfig]] = {
    DetectorType.YOLOv5: YOLOv5ModelConfig,
    DetectorType.YOLOv11: YOLOv5ModelConfig,
}


def get_detector_config(
    model_type: Union[DetectorType, str], **kwargs
) -> BaseModelConfig:
    """Return the appropriate config class for the given detector type."""
    if isinstance(model_type, DetectorType):
        key = model_type
    else:
        key = model_type

    config_cls = DETECTOR_CONFIG_REGISTRY[key]

    return config_cls(**kwargs)
