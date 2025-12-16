import numpy as np


def get_bbox_with_padding(
    coords_px: np.ndarray, padding: float, min_padding_px: int, frame_shape: tuple
):
    """Get bounding box coordinates with padding given pixel coordinates."""
    x_min = int(coords_px[:, 0].min())
    x_max = int(coords_px[:, 0].max())
    y_min = int(coords_px[:, 1].min())
    y_max = int(coords_px[:, 1].max())

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_pad = max(int(np.ceil(padding * max(1, x_range))), min_padding_px)
    y_pad = max(int(np.ceil(padding * max(1, y_range))), min_padding_px)

    x_start = max(x_min - x_pad, 0)
    y_start = max(y_min - y_pad, 0)
    x_stop = min(x_max + x_pad + 1, frame_shape[-1])
    y_stop = min(y_max + y_pad + 1, frame_shape[-2])

    return x_start, y_start, x_stop, y_stop
