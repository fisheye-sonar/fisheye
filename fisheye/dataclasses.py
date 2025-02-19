from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Union

import torch

from fisheye.models.base import BaseModel

BASE = Path(__file__).parent
BEAM_WIDTH_DIR = (BASE / "beam_widths").resolve()


@dataclass
class BaseDatasetConfig:
    """Base dataset configuration."""

    annotations_file: str = None
    beam_width_dir: Path = BEAM_WIDTH_DIR
    batch_size: int = 32
    xdim: int = 0
    ydim: int = 0
    rank: int = -1
    world_size: int = 1
    workers: int = 0
    disable_output: bool = False
    cache_bg_frames: bool = False
    do_bg_subtract: bool = True
    start_frame: int = None
    end_frame: int = None
    num_frames_bg_subtract: int = 1000
    return_unwarped: bool = False
    return_echogram: bool = False
    echogram_filter_kernel: int = 0
    echogram_filter_tol: int = 0.15


@dataclass
class ARISDatasetConfig(BaseDatasetConfig):
    """ARIS dataset configuration."""

    filepath: str = ""


@dataclass
class YOLODatasetConfig(ARISDatasetConfig):
    """YOLO dataset configuration."""

    stride: int = 64
    pad: float = 0.5
    img_size: int = 896


@dataclass
class BaseModelConfig:
    """Base model configuration."""

    model: str | BaseModel = None
    device: str = "cpu"


@dataclass
class YOLOv5ModelConfig(BaseModelConfig):
    """YOLOv5 model config.

    Exposing class variables from YOLOv5's AutoShape.
    """

    conf: float = 0.05  # NMS confidence threshold
    iou: float = 0.2  # NMS IoU threshold
    agnostic: bool = False  # NMS class-agnostic
    multi_label: bool = False  # NMS multiple labels per box
    classes: Optional[list[int]] = (
        None  # (Optional list) filter by class, i.e. = [0, 15, 16] for COCO
    )
    max_det: int = 300  # Maximum number of detections per image
    amp: bool = False  # Automatic Mixed Precision (AMP) inference


@dataclass
class ObjectDetectionPipelineOutput:
    """Object detection pipeline config ."""

    pred_bboxes: Union[torch.Tensor, List[torch.tensor]] = None
    image_shape: List = None
    width: int = None
    height: int = None
