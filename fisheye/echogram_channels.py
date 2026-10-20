"""Channel defaults and validation shared by configs and loaders."""

from collections.abc import Sequence

from fisheye.enums import EchogramChannel

DEFAULT_ECHOGRAM_CHANNELS = [
    EchogramChannel.BGS,
    EchogramChannel.BGS_ANGLE,
]
BGS_ECHOGRAM_CHANNELS = {
    EchogramChannel.BGS,
    EchogramChannel.BGS_ANGLE,
}


def normalize_echogram_channels(
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
