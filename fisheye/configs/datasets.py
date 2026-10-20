from dataclasses import dataclass, field
from pathlib import Path

from fisheye.enums import EchogramChannel

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
    return_frames: bool = True  # Return warped frames
    return_unwarped: bool = False
    return_echogram: bool = False
    echogram_channels: list[EchogramChannel] = field(
        default_factory=lambda: [
            EchogramChannel.BGS,
            EchogramChannel.BGS_ANGLE,
            EchogramChannel.RAW,
        ]
    )
    use_multithreading: bool = True  # For dataloader threading
    max_workers: int = 2
    use_blur: bool = True  # For background subtraction blurring
    return_original_image: bool = False

    def __post_init__(self):
        self.echogram_channels = _coerce_echogram_channels(self.echogram_channels)


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


@dataclass
class YOLODatasetConfig(BaseDatasetConfig):
    """YOLO dataset configuration."""

    stride: int = 64
    pad: float = 0.5
    img_size: int = 896


def _coerce_echogram_channels(channels) -> list[EchogramChannel]:
    if channels is None:
        return [
            EchogramChannel.BGS,
            EchogramChannel.BGS_ANGLE,
            EchogramChannel.RAW,
        ]

    normalized_channels = []
    for index, channel in enumerate(channels):
        if channel is None:
            if index != len(channels) - 1:
                raise ValueError(
                    "None is only supported as the last echogram channel entry"
                )
            break
        try:
            normalized_channels.append(EchogramChannel(channel))
        except ValueError as exc:
            valid_channels = [member.value for member in EchogramChannel]
            raise ValueError(
                f"Unsupported echogram channel {channel!r}. "
                f"Expected one of {valid_channels} or None."
            ) from exc

    if not normalized_channels:
        raise ValueError("echogram_channels must include at least one active channel")

    return normalized_channels
