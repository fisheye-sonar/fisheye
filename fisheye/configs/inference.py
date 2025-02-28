from dataclasses import dataclass, field
from typing import List, Union, TypeVar, Generic

import torch

from fisheye.configs.models import BaseModelConfig, YOLOv5ModelConfig

T = TypeVar("T", bound=BaseModelConfig)


@dataclass
class FishSizeConfig:
    """Configuration for fish size in detection and tracking."""

    min_length: float = 0.3  # Minimum fish length in meters
    max_length: float = 0  # Maximum fish length in meters


@dataclass
class NMSConfig:
    """Non-Maximum Suppression (NMS) configuration."""

    iou: float = 0.25
    # low_conf: float = 0.1
    # high_conf: float = 0.3
    conf: float = 0.1
    max_det: int = 300
    fish_size: FishSizeConfig = FishSizeConfig()


@dataclass
class ObjectDetectionConfig(Generic[T]):
    """Objection detection configuration."""

    model: T = field(default_factory=YOLOv5ModelConfig)
    conf: float = 0.05  # Confidence threshold for detections
    nms_config: NMSConfig = field(default_factory=NMSConfig)
    fish_size: FishSizeConfig = field(default_factory=FishSizeConfig)


@dataclass
class ObjectDetectionPipelineOutput:
    """Object detection pipeline config ."""

    pred_bboxes: Union[torch.Tensor, List[torch.tensor]] = None
    image_shape: List = None
    width: int = None
    height: int = None


@dataclass
class InferenceConfig:
    """Inference configuration."""

    detection: ObjectDetectionConfig = ObjectDetectionConfig()
