from dataclasses import dataclass
from pathlib import Path

from fisheye.enums import EchogramChannel
from fisheye.echogram_channels import normalize_echogram_channels

BASE_DIR = Path(__file__).resolve().parent.parent
BEAM_WIDTH_DIR = BASE_DIR / "beam_widths"


@dataclass
class BaseDatasetConfig:
    """Base dataset configuration.

    Defaults are optimized for running on MPS device.
    """

    filepath: str = ""
    batch_size: int = 16
    rank: int = -1
    world_size: int = 1
    workers: int = 0  # for multiprocessing in dataloader
    cache_bg_frames: bool = False
    do_bg_subtract: bool = True
    start_frame: int = 0  # Default to first frame (zero-indexed)
    end_frame: int = 0
    num_frames_bg_subtract: int = 1000
    # None retains main's primary/auxiliary output contract. A bool opts into
    # independent warped/unwarped/echogram outputs.
    return_frames: bool | None = None
    return_unwarped: bool = False
    return_echogram: bool = False
    return_echogram_with_bg_subtracted: bool = True
    echogram_channels: list[EchogramChannel | str | None] | None = None
    use_multithreading: bool = True  # For dataloader threading
    max_workers: int = 2
    use_blur: bool = True  # For background subtraction blurring
    return_original_image: bool = False

    def __post_init__(self):
        if self.batch_size <= 0 or self.num_frames_bg_subtract <= 0:
            raise ValueError("batch_size and num_frames_bg_subtract must be positive")
        if self.echogram_channels is not None:
            self.echogram_channels = normalize_echogram_channels(self.echogram_channels)


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
    windowlength: float
    samplesperbeam: float
    BeamCount: int
    thesystemtype: int
    numframes: int
    largelens: int
    unwarped_shape: tuple
    beam_width_data: dict
    cycleperiod: float = 0.0
    framerate: float = 0.0


@dataclass
class YOLODatasetConfig(BaseDatasetConfig):
    """YOLO dataset configuration."""

    stride: int = 64
    pad: float = 0.5
    img_size: int = 896
