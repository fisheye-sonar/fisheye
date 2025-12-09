from dataclasses import dataclass, field
from typing import Optional, List

from fisheye.enums import DeviceType, DetectorType, LengthEstimatorType


@dataclass
class BaseModelConfig:
    """Base model configuration."""

    type: str = field(init=False)
    weights: str = None
    device: str = DeviceType.CPU.value


@dataclass
class BaseLengthModelConfig(BaseModelConfig):
    """Base configuration for length estimation models."""

    input_channels: int = 3
    crop_after_model: bool = True
    padd_for_receptive_field: int = 100
    additional_bbox_padding_px: int = 25


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
    """YOLOv11 model config."""

    type: str = DetectorType.YOLOv11.value


@dataclass
class UNetLengthModelConfig(BaseLengthModelConfig):
    """UNet-specific length model configuration."""

    type: str = LengthEstimatorType.UNET.value
    unet_double_conv: bool = False


@dataclass
class HeatmapCNNLengthModelConfig(BaseLengthModelConfig):
    """HeatmapCNN-specific length model configuration."""

    type: str = LengthEstimatorType.HEATMAP_CNN.value
