from dataclasses import dataclass, field
from typing import List, Union, TypeVar, Generic, Dict

import torch

from fisheye.configs.models import BaseModelConfig, YOLOv5ModelConfig
from fisheye.enums import TrackingMethod

T = TypeVar("T", bound=BaseModelConfig)


@dataclass
class TrackerConfig:
    """Configuration for tracking."""

    type: TrackingMethod = TrackingMethod.BYTETRACK
    max_age: int = 20
    min_hits: int = 2
    min_travel: int = 0
    iou_threshold: float = 0.001
    reverse: bool = False


@dataclass
class LengthConfig:
    """Configuration for fish length estimation."""

    min_edge_dist_tolerance_px: int = 10
    vel_delta_tolerance: int = 15
    length_delta_tolerance_cm: int = 5
    vel_window_size: int = 7
    length_window_size: int = 7


@dataclass
class LengthModelConfig:
    """Configuration for the length estimation model."""

    model_type: str = "unet"
    input_channels: int = 3
    unet_double_conv: bool = False
    weights_path: str = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    crop_after_model: bool = True
    padd_for_receptive_field: int = 100
    additional_bbox_padding_px: int = 25


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
    fish_size: FishSizeConfig = field(default_factory=FishSizeConfig)


@dataclass
class ObjectDetectionConfig(Generic[T]):
    """Objection detection configuration."""

    model: T = field(default_factory=YOLOv5ModelConfig)
    conf: float = 0.05  # Confidence threshold for detections
    use_multithreading: bool = True  # Multithreading for model inference
    max_workers: int = 2
    apply_nms_batchwise: bool = False  # Apply NMS batchwise for detection
    apply_length_estimates_batchwise: bool = (
        True  # Apply length estimates batchwise for detection
    )
    nms_config: NMSConfig = field(default_factory=NMSConfig)
    fish_size: FishSizeConfig = field(default_factory=FishSizeConfig)
    length_config: LengthModelConfig = field(default_factory=LengthModelConfig)


@dataclass
class ObjectDetectionPipelineOutput:
    """Object detection pipeline config ."""

    pred_bboxes: Union[torch.Tensor, List[torch.tensor]] = None
    image_shape: List = None
    width: int = None
    height: int = None


@dataclass
class TrackedFish:
    """Metadata for tracked fish."""

    id: int
    bbox: List[float]
    conf: float
    # bbox_index: int
    # det_index: int


@dataclass
class TrackedFrame:
    """Frame-level metadata."""

    frame_num: int
    fish: List[TrackedFish] = field(default_factory=list)


@dataclass
class TrackerOutput:
    """Tracking output."""

    start_frame: int
    end_frame: int
    image_meter_width: float
    image_meter_height: float
    frames: List[TrackedFrame] = field(default_factory=list)
    metadata: List[Dict] = field(default_factory=list)

    @staticmethod
    def dict_to_dataclass(data: Dict):
        frames = [
            TrackedFrame(
                frame_num=frame["frame_num"],
                fish=[
                    TrackedFish(
                        id=fish["fish_id"],
                        bbox=fish["bbox"],
                        conf=fish["conf"],
                        # bbox_index=fish["bbox_index"],
                        # det_index=fish["det_index"],
                    )
                    for fish in frame["fish"]
                ],
            )
            for frame in data["frames"]
        ]

        return TrackerOutput(
            start_frame=data["start_frame"],
            end_frame=data["end_frame"],
            image_meter_width=data["image_meter_width"],
            image_meter_height=data["image_meter_height"],
            frames=frames,
            metadata=data["fish"],
        )
