from dataclasses import dataclass
from pathlib import Path


BASE = Path(__file__).parent.parent
BEAM_WIDTH_DIR = (BASE / "beam_widths").resolve()


@dataclass
class BaseDatasetConfig:
    """Base dataset configuration."""

    beam_width_dir: Path = BEAM_WIDTH_DIR
    batch_size: int = 32
    xdim: int = 0
    ydim: int = 0
    image_meter_width: int = 0
    image_meter_height: int = 0
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


@dataclass
class ARISDatasetConfig(BaseDatasetConfig):
    """ARIS dataset configuration."""

    filepath: str = ""
    image_meter_width = None
    image_meter_height = None


@dataclass
class YOLODatasetConfig(ARISDatasetConfig):
    """YOLO dataset configuration."""

    stride: int = 64
    pad: float = 0.5
    img_size: int = 896
