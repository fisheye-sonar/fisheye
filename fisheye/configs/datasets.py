from dataclasses import dataclass
from pathlib import Path


BASE = Path(__file__).parent.parent
BEAM_WIDTH_DIR = (BASE / "beam_widths").resolve()


@dataclass
class BaseDatasetConfig:
    """Base dataset configuration."""

    beam_width_dir: Path = BEAM_WIDTH_DIR
    filepath: str = ""
    batch_size: int = 32
    rank: int = -1
    world_size: int = 1
    workers: int = 0
    cache_bg_frames: bool = False
    do_bg_subtract: bool = True
    start_frame: int = 0  # Default to first frame (zero-indexed)
    end_frame: int = 0
    num_frames_bg_subtract: int = 1000
    return_unwarped: bool = False
    return_echogram: bool = False
    dev_load_all_frames: bool = (
        True  # Flag to load all frames if end frame and num of frames from header file is 0
    )


@dataclass
class ARISMetadata:
    """Metadata extracted from the ARIS file header.

    Using the same variable names from DIDSON.
    """

    xdim: int
    ydim: int
    image_meter_width: int
    image_meter_height: int
    pixel_meter_size: float
    x_meter_start: float
    x_meter_stop: float
    y_meter_start: float
    y_meter_stop: float
    sampleperiod: float
    soundspeed: float
    windowstart: float
    samplesperbeam: float
    BeamCount: int
    thesystemtype: int
    numframes: int
    largelens: int


@dataclass
class YOLODatasetConfig(BaseDatasetConfig):
    """YOLO dataset configuration."""

    stride: int = 64
    pad: float = 0.5
    img_size: int = 896
