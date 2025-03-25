import json
from collections import defaultdict
from copy import deepcopy

import numpy as np

from fisheye.boxes import iou_batch
from fisheye.configs.inference import TrackerConfig


def linear_assignment(cost_matrix):
    """Note: This implementation is adapted from [SORT](https://github.com/abewley/sort)."""
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate_detections_to_trackers(
    detections, trackers, iou_threshold=TrackerConfig.iou_threshold
):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    Note: This implementation is adapted from [SORT](https://github.com/abewley/sort).
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, _ in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, _ in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class FishMetrics:
    @staticmethod
    def mean_length(tracks, constant, aux=-1):
        return [np.mean(track[2] - track[0]) * constant for track in tracks]

    @staticmethod
    def quantile_length(tracks, constant, aux=-1):
        return [np.quantile(track[2] - track[0], aux) * constant for track in tracks]

    @staticmethod
    def quantile_diagonal(tracks, constant, aux=-1):
        return [
            np.quantile(
                np.sqrt((track[2] - track[0]) ** 2 + (track[3] - track[1]) ** 2), aux
            )
            * constant
            for track in tracks
        ]

    @staticmethod
    def add_lengths(
        json_data,
        length_fn=quantile_length.__func__,
        constant=0.8348286633599985,
        aux=0.8773333335319834,
        output_path=None,
    ):
        json_data = deepcopy(json_data)

        tracks = defaultdict(list)
        for frame in json_data["frames"]:
            for frame_entry in frame["fish"]:
                tracks[frame_entry["fish_id"]].append(np.array(frame_entry["bbox"]))
        tracks = [np.array(track).T for _, track in sorted(tracks.items())]

        lengths = np.array(
            length_fn(tracks, constant * json_data["image_meter_width"], aux=aux)
        )

        for fish, fish_length in zip(
            sorted(json_data["fish"], key=lambda k: k["id"]), lengths
        ):
            fish["length"] = fish_length

        if output_path is not None:
            with open(output_path, "w") as output:
                json.dump(json_data, output, indent=2)

        return json_data

    @staticmethod
    def get_direction(start_bbox, end_bbox):
        start_center, end_center = (start_bbox[2] + start_bbox[0]) / 2, (
            end_bbox[2] + end_bbox[0]
        ) / 2
        return (
            "right"
            if start_center < 0.5 <= end_center
            else "left" if start_center >= 0.5 > end_center else "none"
        )

    @staticmethod
    def get_travel_distance(
        start_bbox, end_bbox, image_meter_width, image_meter_height
    ):
        dx, dy = (
            (start_bbox[2] + start_bbox[0]) / 2 - (end_bbox[2] + end_bbox[0]) / 2
        ) * image_meter_width, (
            (start_bbox[3] + start_bbox[1]) / 2 - (end_bbox[3] + end_bbox[1]) / 2
        ) * image_meter_height

        return np.sqrt(dx**2 + dy**2)
