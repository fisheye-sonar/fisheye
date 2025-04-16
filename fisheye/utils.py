from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def torch_distributed_zero_first(rank):
    if rank != -1:
        torch.distributed.barrier()
    yield
    if rank != -1:
        torch.distributed.barrier()


def yolo_collate_fn(batch):
    """See ScaledYOLOv4.utils.datasets.collate_fn"""

    img, label, shapes = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0), shapes


n_points = 6


def calculate_unwarped_points(points, info, xdim, ydim):
    """
    points: [..., 2] in x,y
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
        [info["x_meter_start"] + i * info["pixel_meter_size"] for i in range(xdim)]
    )
    y_meter_values = np.array(
        [info["y_meter_start"] - i * info["pixel_meter_size"] for i in range(ydim)]
    )
    beam_edges = info["beam_width_data"]["beam_left"].to_numpy()
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
    bin_length = info["sampleperiod"] * 0.000001 * info["soundspeed"] / 2.0
    unwarped_points = []

    for point in points:
        xa = point[0]
        ya = point[1]

        angle = np.arctan(x_meter_values[xa] / y_meter_values[ya])
        rad = (x_meter_values[xa] ** 2 + y_meter_values[ya] ** 2) ** 0.5
        rad_pix = rad / info["pixel_meter_size"]

        beam_num = np.digitize(np.rad2deg(angle), beam_edges_expanded) - 1
        beam_num -= len(beam_left_pad)

        dist_from_bottom = rad - info["windowstart"]

        bin_num = info["samplesperbeam"] - (dist_from_bottom / bin_length).astype(int)
        unwarped_coord = [beam_num, bin_num]
        unwarped_points.append(unwarped_coord)

    unwarped_points = np.stack(unwarped_points)
    unwarped_points = unwarped_points.reshape(input_shape)

    return unwarped_points
