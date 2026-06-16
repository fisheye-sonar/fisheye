from collections.abc import Sequence

import numpy as np

from fisheye.enums import EchogramChannel

DEFAULT_ECHOGRAM_CHANNELS = [
    EchogramChannel.BGS,
    EchogramChannel.BGS_ANGLE,
    EchogramChannel.RAW,
]
BGS_ECHOGRAM_CHANNELS = {
    EchogramChannel.BGS,
    EchogramChannel.BGS_ANGLE,
}


def _normalize_echogram_channels(
    echogram_channels: Sequence[EchogramChannel | str | None] | None,
) -> list[EchogramChannel]:
    """Validate channel selection and trim an optional trailing None."""
    if echogram_channels is None:
        return DEFAULT_ECHOGRAM_CHANNELS.copy()

    normalized_channels = []
    for index, channel in enumerate(echogram_channels):
        if channel is None:
            if index != len(echogram_channels) - 1:
                raise ValueError(
                    "None is only supported as the last echogram channel entry"
                )
            break
        try:
            channel = EchogramChannel(channel)
        except ValueError as exc:
            valid_channels = [member.value for member in EchogramChannel]
            raise ValueError(
                f"Unsupported echogram channel {channel!r}. "
                f"Expected one of {valid_channels} or None."
            ) from exc
        normalized_channels.append(channel)

    if not normalized_channels:
        raise ValueError("echogram_channels must include at least one active channel")

    return normalized_channels


def echogram_uses_bg_subtraction(
    echogram_channels: Sequence[EchogramChannel | str | None] | None,
) -> bool:
    """Return True when any selected channel requires background subtraction."""
    normalized_channels = _normalize_echogram_channels(echogram_channels)
    return any(channel in BGS_ECHOGRAM_CHANNELS for channel in normalized_channels)


def _normalize_center_line(center_line_echogram: np.ndarray) -> np.ndarray:
    """Clamp the center-line channel into a stable [0, 1] range."""
    center_line_echogram = np.clip(center_line_echogram, 0, 1)
    max_value = np.max(center_line_echogram)
    if max_value > 0:
        center_line_echogram = center_line_echogram / max_value
    return center_line_echogram.astype(np.float32, copy=False)


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

    echogram_channels = _normalize_echogram_channels(echogram_channels)
    do_bg_subtract_echogram = echogram_uses_bg_subtraction(echogram_channels)

    output = np.zeros(
        (*unwarped_frames.shape[:2], len(echogram_channels)),
        dtype=np.float32,
    )
    depth = unwarped_frames.shape[2]

    frames_f32 = unwarped_frames.astype(np.float32)

    raw_echogram = np.max(frames_f32, axis=2) / 255.0

    bgs_angle_echogram = None
    if do_bg_subtract_echogram:
        bgs_frames = frames_f32
        if mean_blurred_frame is None or mean_normalization_value is None:
            raise ValueError(
                "mean_blurred_frame and mean_normalization_value are required when "
                "do_bg_subtract_echogram=True"
            )
        bgs_frames = bgs_frames - mean_blurred_frame
        bgs_frames = bgs_frames / mean_normalization_value
        bgs_echogram = np.max(bgs_frames, axis=2).astype(np.float32, copy=False)
        if EchogramChannel.BGS_ANGLE in echogram_channels:
            bgs_angle_echogram = np.argmax(bgs_frames, axis=2).astype(
                np.float32
            ) / float(depth)
            bgs_angle_echogram -= 0.5
        del bgs_frames
    else:
        bgs_echogram = None

    for channel_index, channel_name in enumerate(echogram_channels):
        if channel_name == EchogramChannel.ZERO:
            continue
        elif channel_name == EchogramChannel.RAW:
            output[:, :, channel_index] = raw_echogram
        elif channel_name == EchogramChannel.BGS:
            output[:, :, channel_index] = bgs_echogram
        elif channel_name == EchogramChannel.BGS_ANGLE:
            output[:, :, channel_index] = bgs_angle_echogram
        elif channel_name == EchogramChannel.ANGLE:
            output[:, :, channel_index] = (
                np.argmax(frames_f32, axis=2).astype(np.float32) / float(depth)
            ) - 0.5
        elif channel_name == EchogramChannel.CENTER_LINE:
            center_line_echgram = frames_f32[:, :, frames_f32.shape[2] // 2]
            output[:, :, channel_index] = _normalize_center_line(center_line_echgram)
        else:
            raise ValueError(f"Unsupported echogram channel {channel_name!r}")

    return output
