from dataclasses import dataclass
from typing import Optional

from fisheye.detect.base import BaseModel
from fisheye.enums import DeviceType


@dataclass
class BaseModelConfig:
    """Base model configuration."""

    weights: str | BaseModel = None
    device: str = DeviceType.MPS.value


@dataclass
class YOLOv5ModelConfig(BaseModelConfig):
    """YOLOv5 model config.

    Exposing class variables from YOLOv5's AutoShape.
    """

    agnostic: bool = False  # NMS class-agnostic
    multi_label: bool = False  # NMS multiple labels per box
    classes: Optional[list[int]] = (
        None  # (Optional list) filter by class, i.e. = [0, 15, 16] for COCO
    )
    max_det: int = 300  # Maximum number of detections per image
    amp: bool = False  # Automatic Mixed Precision (AMP) inference
