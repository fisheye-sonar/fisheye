from dataclasses import dataclass, field
from typing import List, Union, TypeVar, Generic

import torch

from fisheye.configs.models import BaseModelConfig, YOLOv5ModelConfig
from fisheye.enums import TrackingMethod

T = TypeVar("T", bound=BaseModelConfig)


@dataclass
class TrackerConfig:
    """Configuration for tracking."""

    type: TrackingMethod = TrackingMethod.BYTETRACK
    max_age: int = 0
    min_hits: int = 3
    min_travel: int = 0
    iou_threshold: float = 0.05
    reverse: bool = False


@dataclass
class FishSizeConfig:
    """Configuration for fish size in detection and tracking."""

    min_length: float = 0.3  # Minimum fish length in meters
    max_length: float = 0  # Maximum fish length in meters


@dataclass
class NMSConfig:
    """Non-Maximum Suppression (NMS) configuration."""

    iou: float = 0.25  # NMS iou score
    conf: float = 0.1  # NMS confidence score
    max_det: int = 300  # Maximum number of detections
    max_nms: int = 30000  # Maximum number of boxes into torchvision.ops.nms()
    redundant: bool = True  # Require redundant detections
    merge: bool = False  # Use merge-NMS
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
