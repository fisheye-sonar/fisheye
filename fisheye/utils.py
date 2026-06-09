from typing import List, Union

import numpy as np
import pandas as pd

from fisheye.configs.datasets import BEAM_WIDTH_DIR, ARISMetadata
from fisheye.dataloaders.didson import pyARIS


def calculate_unwarped_points(points, info, xdim, ydim):
    """
    points: [..., 2] in x,y
    info: ARISMetdata dataclass
    output: [..., 2] in x,y unwarped (technically theta, R)

    NOTE:   1.The extra beams are extrapolated off the first and last beam spacing, but the beams are not evenly
    spaced so this is approximate 2.We calculate the extrapolate the beam number off the left and right for half the
    total sensor width, if the point is very far off the right or left of the beam this will be incorrect
    """
    if len(points) == 0:
        return np.array((0, 4))
    if isinstance(points, list):
        points = np.array(points)
    input_shape = points.shape
    points = points.reshape((-1, 2))
    points = points.astype(int)
    x_meter_values = np.array(
        [info.x_meter_start + i * info.pixel_meter_size for i in range(xdim)]
    )
    y_meter_values = np.array(
        [info.y_meter_start - i * info.pixel_meter_size for i in range(ydim)]
    )
    beam_edges = info.beam_width_data["beam_left"].to_numpy()
    beam_edges_pad = len(beam_edges) // 2
    beam_left_diff = beam_edges[1] - beam_edges[0]
    beam_right_diff = beam_edges[-1] - beam_edges[-2]

    beam_left_pad = (
        np.arange(-beam_edges_pad * beam_left_diff, 0, beam_left_diff) + beam_edges[0]
    )
    beam_right_pad = (
        np.arange(
            beam_right_diff,
            beam_edges_pad * beam_right_diff,
            beam_right_diff,
        )
        + beam_edges[-1]
    )

    beam_edges_expanded = np.concatenate((beam_left_pad, beam_edges, beam_right_pad), 0)
    bin_length = info.sampleperiod * 0.000001 * info.soundspeed / 2.0
    unwarped_points = []

    for point in points:
        xa = point[0]
        ya = point[1]

        angle = np.arctan(x_meter_values[xa] / y_meter_values[ya])
        rad = (x_meter_values[xa] ** 2 + y_meter_values[ya] ** 2) ** 0.5
        rad_pix = rad / info.pixel_meter_size

        beam_num = np.digitize(np.rad2deg(angle), beam_edges_expanded) - 1
        beam_num -= len(beam_left_pad)

        dist_from_bottom = rad - info.windowstart

        bin_num = info.samplesperbeam - (dist_from_bottom / bin_length).astype(int)
        unwarped_coord = [beam_num, bin_num]
        unwarped_points.append(unwarped_coord)

    unwarped_points = np.stack(unwarped_points)
    unwarped_points = unwarped_points.reshape(input_shape)

    return unwarped_points


def calculate_warped_points(points, info, xdim, ydim):
    """
    Calculate the warped points.
    Args:
        points: [..., 2] in unwarped coordinates
        info: Info dict pulled from ARIS header.
        xdim: Width of the image in pixels.
        ydim: Height of the image in pixels.
    Returns:
        warped_points: [..., 2] in x,y warped
    """
    if len(points) == 0:
        return np.array((0, 4))
    if isinstance(points, list):
        points = np.array(points)

    input_shape = points.shape
    points = points.reshape((-1, 2))
    points = points.astype(int)

    bin_length = info.sampleperiod * 0.000001 * info.soundspeed / 2.0

    warped_points_xy = []
    for point in points:
        beam_num = point[0]
        bin_num = info.samplesperbeam - point[1]
        bin_front_edge_distance = info.windowstart + bin_length * bin_num
        bin_back_edge_distance = info.windowstart + bin_length * (bin_num + 1)

        df = pd.DataFrame(info.beam_width_data)
        beam_angles = df[df["beam_num"] == beam_num]
        a1 = beam_angles["beam_left"].iloc[0]
        a2 = beam_angles["beam_right"].iloc[0]

        # TODO (MAH): I can't figure out whats going on with the beam spacing in the files.
        # Once the center point crosses 0, the ordering of the left and right angles swap...
        # For now I'll assume the y axis is the common line. Positive angles go to the left,
        # negative angles go to the right
        left = max(a1, a2)
        right = min(a1, a2)

        # Left Edge
        beam_left_angle = np.deg2rad(left)
        # MAH 2025-03-05 18:42:23 the sign of the sin had to be flipped to make it in the right place
        rot_matrix = np.array(
            [
                [np.cos(beam_left_angle), np.sin(beam_left_angle)],
                [-np.sin(beam_left_angle), np.cos(beam_left_angle)],
            ]
        )
        vec = np.array([0, bin_back_edge_distance])
        bin_left_back_point = np.matmul(rot_matrix, vec)

        vec = np.array([0, bin_front_edge_distance])
        bin_left_front_point = np.matmul(rot_matrix, vec)

        # Right Edge
        beam_right_angle = np.deg2rad(right)
        rot_matrix = np.array(
            [
                [np.cos(beam_right_angle), np.sin(beam_right_angle)],
                [-np.sin(beam_right_angle), np.cos(beam_right_angle)],
            ]
        )

        vec = np.array([0, bin_front_edge_distance])
        bin_right_front_point = np.matmul(rot_matrix, vec)

        vec = np.array([0, bin_back_edge_distance])
        bin_right_back_point = np.matmul(rot_matrix, vec)

        bl, br, fr, fl = (
            bin_left_back_point,
            bin_right_back_point,
            bin_right_front_point,
            bin_left_front_point,
        )
        # Determine the center of the sample box
        center_x = (fl[0] + fr[0]) / 2.0
        center_y = (bl[1] + fl[1]) / 2.0

        # Convert the center coordinates to pixel indices
        i = int((info.y_meter_start - center_y) / info.pixel_meter_size)
        j = int((center_x - info.x_meter_start) / info.pixel_meter_size)

        # Ensure the indices are within the image bounds
        if i < 0 or i >= ydim or j < 0 or j >= xdim:
            return None, None

        warped_points_xy.append([j, i])

    warped_points_xy = np.stack(warped_points_xy)
    warped_points_xy = warped_points_xy.reshape(input_shape)

    return warped_points_xy


def get_unwarped_distance_and_theta(row: pd.Series):
    """Get the distance (range) and beam angle to a detection.

    Args:
        row (pd.Series): A DataFrame row with keys:
            - "bbox": Bounding box given as [x_center, y_center, width, height] relative to the original image space
            - "metadata": Object with ARIS header fields (e.g., dimensions, pixel size)

    Returns:
        tuple: (distance: float, theta: float)
            - distance: Distance (range) to the unwarped point in meters
            - theta: Beam center angle in degrees
    """
    metadata = row["metadata"]

    if metadata.beam_width_data is None:
        metadata.beam_width_data, _ = pyARIS.load_beam_width_data(
            frame=metadata, beam_width_dir=BEAM_WIDTH_DIR
        )

    bbox_xywh = np.array(row["bbox"]) * np.array(
        [metadata.xdim, metadata.ydim, metadata.xdim, metadata.ydim]
    )
    x_center_px = bbox_xywh[0]
    y_center_px = bbox_xywh[1]

    points_xy_unwarped = calculate_unwarped_points(
        [x_center_px, y_center_px], metadata, metadata.xdim, metadata.ydim
    )
    beam_num = min(points_xy_unwarped[0], metadata.BeamCount - 1)
    bin_num = points_xy_unwarped[1]

    bin_length = metadata.sampleperiod * 0.000001 * metadata.soundspeed / 2.0
    distance = round(
        float(metadata.windowstart + (metadata.samplesperbeam - bin_num) * bin_length),
        2,
    )

    theta = round(
        float(metadata.beam_width_data.iloc[beam_num]["beam_center"]),
        2,
    )

    return distance, theta


def convert_pixels_to_coords_meters(
    coords_px: Union[np.ndarray, List], metadata: ARISMetadata
) -> np.ndarray:
    """Convert image pixel coords to ARIS world (meter) coords.

    Args:
        coords_px: [N, 2] array of (x, y) pixel coordinates
        metadata: ARISMetadata object containing spatial configuration

    Returns:
        [N, 2] array of (x, y) meter coordinates
    """
    if isinstance(coords_px, list):
        coords_px = np.array(coords_px)

    x_aris_max = metadata.x_meter_stop
    x_aris_min = metadata.x_meter_start
    y_aris_max = metadata.y_meter_start
    y_aris_min = metadata.y_meter_stop

    xdim = int(metadata.xdim)
    ydim = int(metadata.ydim)

    coords_m = coords_px.copy().astype(float)
    coords_m[:, 0] = (coords_px[:, 0] / xdim) * (x_aris_max - x_aris_min) + x_aris_min
    coords_m[:, 1] = y_aris_max - (coords_px[:, 1] / ydim) * (y_aris_max - y_aris_min)

    return coords_m


def convert_coords_meters_to_pixels(coords: np.ndarray, metadata: dict):
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
