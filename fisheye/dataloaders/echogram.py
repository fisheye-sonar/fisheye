from collections.abc import Sequence

import numpy as np

from fisheye.enums import EchogramChannel

from fisheye.echogram_channels import (
    BGS_ECHOGRAM_CHANNELS,
    normalize_echogram_channels,
)


def echogram_uses_bg_subtraction(
    echogram_channels: Sequence[EchogramChannel | str | None] | None,
) -> bool:
    """Return True when any selected channel requires background subtraction."""
    normalized_channels = normalize_echogram_channels(echogram_channels)
    return any(channel in BGS_ECHOGRAM_CHANNELS for channel in normalized_channels)


def _normalize_center_line(center_line_echogram: np.ndarray) -> np.ndarray:
    """Clamp the center-line channel into a stable [0, 1] range."""
    return np.clip(center_line_echogram / 255.0, 0, 1).astype(np.float32, copy=False)


def compute_echogram(
    unwarped_frames: np.ndarray,
    mean_blurred_frame: np.ndarray | None = None,
    mean_normalization_value: float | np.ndarray | None = None,
    echogram_channels: Sequence[EchogramChannel | str | None] | None = None,
) -> np.ndarray:
    """
    Generate an echogram from unwarped beam frames.

    ``echogram_channels`` is ordered and may contain ``"bgs"``,
    ``"bgs_angle"``, ``"angle"``, ``"raw"``, ``"center_line"``, ``"0"``,
    and an optional trailing ``None``.
    """

    echogram_channels = normalize_echogram_channels(echogram_channels)
    do_bg_subtract_echogram = echogram_uses_bg_subtraction(echogram_channels)

    output = np.zeros(
        (*unwarped_frames.shape[:2], len(echogram_channels)),
        dtype=np.float32,
    )
    depth = unwarped_frames.shape[2]

    bgs_echogram = None
    bgs_angle_echogram = None
    if do_bg_subtract_echogram:
        if mean_blurred_frame is None or mean_normalization_value is None:
            raise ValueError(
                "mean_blurred_frame and mean_normalization_value are required when "
                "do_bg_subtract_echogram=True"
            )
        bgs_frames = np.subtract(unwarped_frames, mean_blurred_frame, dtype=np.float32)
        bgs_frames /= np.where(
            np.asarray(mean_normalization_value) > 0, mean_normalization_value, 1.0
        )
        if EchogramChannel.BGS in echogram_channels:
            bgs_echogram = np.max(bgs_frames, axis=2)
        if EchogramChannel.BGS_ANGLE in echogram_channels:
            bgs_angle_echogram = np.argmax(bgs_frames, axis=2).astype(
                np.float32
            ) / float(depth)
            bgs_angle_echogram -= 0.5
        del bgs_frames

    for channel_index, channel_name in enumerate(echogram_channels):
        if channel_name == EchogramChannel.ZERO:
            continue
        elif channel_name == EchogramChannel.RAW:
            output[:, :, channel_index] = np.max(unwarped_frames, axis=2) / 255.0
        elif channel_name == EchogramChannel.BGS:
            output[:, :, channel_index] = bgs_echogram
        elif channel_name == EchogramChannel.BGS_ANGLE:
            output[:, :, channel_index] = bgs_angle_echogram
        elif channel_name == EchogramChannel.ANGLE:
            output[:, :, channel_index] = (
                np.argmax(unwarped_frames, axis=2).astype(np.float32) / float(depth)
            ) - 0.5
        elif channel_name == EchogramChannel.CENTER_LINE:
            center_line_echogram = unwarped_frames[:, :, depth // 2]
            output[:, :, channel_index] = _normalize_center_line(center_line_echogram)
        else:
            raise ValueError(f"Unsupported echogram channel {channel_name!r}")

    return output
