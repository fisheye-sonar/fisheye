import numpy as np

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


def get_unwarped_distance(row):
    """Get the distance to an unwarped point.

    Function requires a DataFrame row where it has bounding boxes [x, y, width, height] and DIDSON header information
    under the column name `metadata`
    """
    metadata = row["metadata"]
    metadata.beam_width_data, _ = pyARIS.load_beam_width_data(
        frame=metadata, beam_width_dir="/Users/madison/Code/fisheye/fisheye/beam_widths"
    )

    bbox_xywh = np.array(row["bbox"]) * np.array(
        [metadata.xdim, metadata.ydim, metadata.xdim, metadata.ydim]
    )
    # average_x, average y
    bbox_xywh = [bbox_xywh[0] + bbox_xywh[2] / 2, bbox_xywh[1] + bbox_xywh[3] / 2, 1, 1]
    bbox_xywh = np.expand_dims(np.array(bbox_xywh), axis=0)

    points_xy = [bbox_xywh[0][0], bbox_xywh[0][1]]
    points_xy_unwarped = calculate_unwarped_points(
        points_xy, metadata, metadata.xdim, metadata.ydim
    )
    distance = (
        metadata.ydim - points_xy_unwarped[1]
    ) * metadata.pixel_meter_size + metadata.y_meter_stop

    return distance
