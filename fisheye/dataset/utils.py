import numpy as np


def get_bbox_with_padding(coords_px, padding, min_padding_px, frame_shape):
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


def coords_meters_to_pixels(coords: np.ndarray, metadata: dict):
    """Convert ARIS world coords to pixel coords."""
    x_aris_max = metadata["x_meter_stop"]
    x_aris_min = metadata["x_meter_start"]
    y_aris_max = metadata["y_meter_start"]
    y_aris_min = metadata["y_meter_stop"]

    xdim = int(metadata["xdim"])
    ydim = int(metadata["ydim"])

    # Convert to pixel space
    x = (xdim) * (coords[:, 0] - x_aris_min) / (x_aris_max - x_aris_min)
    y = (ydim) * (y_aris_max - coords[:, 1]) / (y_aris_max - y_aris_min)

    # Clip to valid pixel ranges
    coords_px = np.rint(np.stack((x, y), axis=1)).astype(np.int64)
    coords_px[:, 0] = np.clip(coords_px[:, 0], 0, xdim - 1)
    coords_px[:, 1] = np.clip(coords_px[:, 1], 0, ydim - 1)

    return coords_px
