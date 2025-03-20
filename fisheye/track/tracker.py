import json
from collections import Counter
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from fisheye.configs.inference import TrackerConfig, FishSizeConfig
from fisheye.enums import TrackingMethod
from fisheye.track.bytetrack import ByteTracker
from fisheye.track.sort import Sort
from fisheye.track.utils import FishMetrics

# Add any new trackers here
TRACKER_CLASSES = {
    TrackingMethod.BYTETRACK: ByteTracker,
    TrackingMethod.SORT: Sort,
}


class Tracker:
    def __init__(
        self,
        clip_info,
        config: TrackerConfig,
    ):
        self.algorithm = self._initialize_tracker(
            config.type,
            {
                "max_age": config.max_age,
                "min_hits": config.min_hits,
                "iou_threshold": config.iou_threshold,
            },
        )
        self.fish_ids = Counter()
        self.reverse = config.reverse
        self.min_hits = config.min_hits
        self.json_data = deepcopy(clip_info)

        if self.reverse:
            self.frame_id = self.json_data["end_frame"]
        else:
            self.frame_id = self.json_data["start_frame"]

        self.json_data["frames"] = []

    @staticmethod
    def _initialize_tracker(algorithm: TrackingMethod, args):
        tracker_cls = TRACKER_CLASSES.get(algorithm)

        if tracker_cls is None:
            raise ValueError(f"Tracking method {algorithm} is not supported.")

        return tracker_cls(**args)

    def update(self, dets=np.empty((0, 5))):
        """Updates the tracker with new detections. Boxes should be given in normalized [x1,y1,x2,y2,c]"""
        new_frame_entries = []
        for track in self.algorithm.update(dets):

            # Match confidence with correct track
            conf = 0
            min_score = 1000000
            if TrackingMethod.SORT == self.algorithm.type:
                conf, min_score = self.algorithm.match_confidence_to_track(
                    track, dets, conf, min_score
                )

            elif TrackingMethod.BYTETRACK == self.algorithm.type:
                # dets[0] = low conf, dets[1] = high conf
                for det_group in dets:
                    conf, min_score = self.algorithm.match_confidence_to_track(
                        track, det_group, conf, min_score
                    )

            # Assign Track
            self.fish_ids[int(track[4])] += 1
            new_frame_entries.append(
                {
                    "fish_id": int(track[4]),
                    "bbox": list(track[:4]),
                    "visible": 1,
                    "human_labeled": 0,
                    "conf": conf,
                }
            )
        new_frame_entries = sorted(new_frame_entries, key=lambda k: k["fish_id"])

        self.json_data["frames"].append(
            {"frame_num": self.frame_id, "fish": new_frame_entries}
        )
        if self.reverse:
            self.frame_id -= 1
        else:
            self.frame_id += 1

    def finalize(
        self,
        min_length,
        min_travel,
        output_path=None,
    ):  # vert_margin=0.0
        json_data = deepcopy(self.json_data)

        # map (valid) fish IDs to 0, 1, 2, ...
        fish_id_map = {}
        for fish_id, count in self.fish_ids.items():
            if count >= self.min_hits:
                fish_id_map[fish_id] = len(fish_id_map)

        # separate frame boxes into tracks, keyed by mapped IDs
        # each track is a list of tuples ( bbox, frame_num )
        tracks = {v: [] for v in fish_id_map.values()}
        for frame in json_data["frames"]:
            for bbox in frame["fish"]:
                # check if valid
                if bbox["fish_id"] in fish_id_map.keys():
                    track_id = fish_id_map[bbox["fish_id"]]
                    tracks[track_id].append((bbox["bbox"], frame["frame_num"]))

        # map IDs and keep frame['fish'] sorted by ID
        for frame in json_data["frames"]:
            new_frame_entries = []
            for frame_entry in frame["fish"]:
                if frame_entry["fish_id"] in fish_id_map:
                    frame_entry["fish_id"] = fish_id_map[frame_entry["fish_id"]]
                    new_frame_entries.append(frame_entry)
            frame["fish"] = sorted(new_frame_entries, key=lambda k: k["fish_id"])

        # create summary 'fish' entry for json data
        json_data["fish"] = []
        for track_id, boxes in tracks.items():
            fish_entry = {}
            fish_entry["id"] = track_id
            fish_entry["length"] = -1

            start_bbox = boxes[0][0]
            end_bbox = boxes[-1][0]
            fish_entry["direction"] = FishMetrics.get_direction(start_bbox, end_bbox)

            fish_entry["travel_dist"] = FishMetrics.get_travel_distance(
                start_bbox,
                end_bbox,
                json_data["image_meter_width"],
                json_data["image_meter_height"],
            )

            fish_entry["start_frame_index"] = boxes[0][1]
            fish_entry["end_frame_index"] = boxes[-1][1]
            fish_entry["color"] = FishMetrics.select_color(track_id)

            json_data["fish"].append(fish_entry)

        # filter 'fish' field by fish length and travel distance
        json_data = FishMetrics.add_lengths(json_data)
        invalid_ids = []
        if min_length != -1.0:
            new_fish = []
            for fish in json_data["fish"]:
                if fish["length"] > min_length and fish["travel_dist"] > min_travel:
                    new_fish.append(fish)
                else:
                    invalid_ids.append(fish["id"])
            json_data["fish"] = new_fish

        # filter 'frames' field by fish length
        if len(invalid_ids):
            for frame in json_data["frames"]:
                new_fish = []
                for fish in frame["fish"]:
                    if fish["fish_id"] not in invalid_ids:
                        new_fish.append(fish)
                frame["fish"] = new_fish

        if output_path is not None:
            with open(output_path, "w") as output:
                json.dump(json_data, output, indent=2)

        return json_data

    def state(self, output_path=None):
        json_data = deepcopy(self.json_data)

        if output_path is not None:
            with open(output_path, "w") as output:
                json.dump(json_data, output, indent=2)

        return json_data


def run_tracker(
    low_preds,
    high_preds,
    image_meter_width,
    image_meter_height,
    tracking_config,
    min_length=FishSizeConfig.min_length,
    gp=None,
    verbose=True,
):
    """Factory method to run tracker."""
    if gp:
        gp(0, f"Tracking using {tracking_config}...")

    clip_info = {
        "start_frame": 0,
        "end_frame": len(low_preds),
        "image_meter_width": image_meter_width,
        "image_meter_height": image_meter_height,
    }

    tracker = Tracker(
        clip_info=clip_info,
        config=tracking_config,
    )

    with tqdm(
        total=len(low_preds),
        desc=f"Running tracker using {tracking_config.type}",
        ncols=0,
        disable=not verbose,
    ) as pbar:
        for i, key in enumerate(
            sorted(low_preds.keys(), reverse=tracking_config.reverse)
        ):
            if gp:
                gp(i / len(low_preds), pbar.__str__())
            low_boxes, high_boxes = low_preds[key], high_preds[key]
            boxes = (
                (low_boxes, high_boxes)
                if low_boxes is not None and high_boxes is not None
                else (np.empty((0, 5)), np.empty((0, 5)))
            )
            tracker.update(boxes)
            pbar.update(1)

    json_data = tracker.finalize(
        min_length=min_length, min_travel=tracking_config.min_travel
    )

    return json_data
