import json
from collections import Counter
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from fisheye.configs.inference import TrackerConfig
from fisheye.enums import TrackingMethod
from fisheye.models.track.bytetrack import ByteTracker
from fisheye.models.track.sort import Sort
from fisheye.models.track.utils import FishMetrics

# Add any new trackers here
TRACKER_CLASSES = {
    TrackingMethod.BYTETRACK: ByteTracker,
    TrackingMethod.SORT: Sort,
}


class Tracker:
    def __init__(
        self,
        clip_info,
        algorithm: TrackingMethod,
        args={
            "max_age": TrackerConfig.max_age,
            "min_hits": TrackerConfig.min_hits,
            "iou_threshold": TrackerConfig.iou_threshold,
        },
        reverse=TrackerConfig.reverse,
    ):

        algorithm = self._initialize_tracker(algorithm, args)
        self.algorithm = algorithm(**args)
        self.fish_ids = Counter()
        self.reverse = reverse
        self.min_hits = args.get("min_hits")
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

    # Boxes should be given in normalized [x1,y1,x2,y2,c]
    def update(self, dets=np.empty((0, 5))):
        new_frame_entries = []
        for track in self.algorithm.update(dets):

            # Match confidence with correct track
            conf = 0
            min_score = 1000000
            if TrackingMethod.SORT == track.type:
                for det in dets:
                    score = sum(abs(det[0:4] - track[0:4]))
                    if score < min_score:
                        min_score = score
                        conf = det[4]

            elif TrackingMethod.BYTETRACK == track.type:
                for det in dets[0]:
                    score = sum(abs(det[0:4] - track[0:4]))
                    if score < min_score:
                        min_score = score
                        conf = det[4]
                for det in dets[1]:
                    score = sum(abs(det[0:4] - track[0:4]))
                    if score < min_score:
                        min_score = score
                        conf = det[4]

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
        self, output_path=None, min_length=-1.0, min_travel=-1.0
    ):  # vert_margin=0.0
        json_data = deepcopy(self.json_data)

        # map (valid) fish IDs to 0, 1, 2, ...
        fish_id_map = {}
        for fish_id, count in self.fish_ids.items():
            if count >= self.min_hits:
                fish_id_map[fish_id] = len(fish_id_map)

        # separate frame boxes into tracks, keyed by mapped IDs
        # each track is a list of tuples ( bbox, frame_num )
        tracks = {v: [] for _, v in fish_id_map.items()}
        for frame in json_data["frames"]:
            for bbox in frame["fish"]:
                # check if valid
                if bbox["fish_id"] in fish_id_map.keys():
                    track_id = fish_id_map[bbox["fish_id"]]
                    tracks[track_id].append((bbox["bbox"], frame["frame_num"]))

        # map IDs and keep frame['fish'] sorted by ID
        for i, frame in enumerate(json_data["frames"]):
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


# class Tracker:
#     def __init__(self, clip_info, config):
#         self.algorithm = config.type(**config)
#         self.fish_ids = Counter()
#         self.reverse = config.reverse
#         self.min_hits = config.min_hits
#         self.json_data = deepcopy(clip_info)
#         self.frame_id = (
#             self.json_data["end_frame"]
#             if self.reverse
#             else self.json_data["start_frame"]
#         )
#
#         self.json_data["frames"] = []
#
#     def run(self, detections=np.empty((0, 5))):
#         """Boxes should be given in normalized [x1,y1,x2,y2,c]"""
#         new_frame_entries = [
#             self._process_track(track, detections)
#             for track in self.algorithm.update(detections)
#         ]
#         self.json_data["frames"].append(
#             {
#                 "frame_num": self.frame_id,
#                 "fish": sorted(new_frame_entries, key=lambda k: k["fish_id"]),
#             }
#         )
#         self.frame_id += -1 if self.reverse else 1
#
#     def finalize(self, output_path=None, min_length=-1.0, min_travel=-1.0):
#         """Process tracking results with valid fish IDs."""
#         json_data = deepcopy(self.json_data)
#         fish_id_map, tracks = self._map_fish_ids(json_data)
#
#         for frame in json_data["frames"]:
#             frame["fish"] = sorted(
#                 [
#                     {**entry, "fish_id": fish_id_map[entry["fish_id"]]}
#                     for entry in frame["fish"]
#                     if entry["fish_id"] in fish_id_map
#                 ],
#                 key=lambda k: k["fish_id"],
#             )
#
#         # Create summary 'fish' entry for json data
#         json_data["fish"] = []
#         for track_id, boxes in tracks.items():
#             fish_entry = {}
#             fish_entry["id"] = track_id
#
#             # TODO (MVH) - why is this hard coded?
#             fish_entry["length"] = -1
#
#             start_bbox = boxes[0][0]
#             end_bbox = boxes[-1][0]
#             fish_entry["direction"] = FishMetrics.get_direction(start_bbox, end_bbox)
#             fish_entry["travel_dist"] = FishMetrics.get_travel_distance(
#                 start_bbox,
#                 end_bbox,
#                 json_data["image_meter_width"],
#                 json_data["image_meter_height"],
#             )
#
#             fish_entry["start_frame_index"] = boxes[0][1]
#             fish_entry["end_frame_index"] = boxes[-1][1]
#             fish_entry["color"] = FishMetrics.select_color(track_id)
#             json_data["fish"].append(fish_entry)
#
#         json_data = FishMetrics.add_lengths(json_data)
#
#         # Filter valid fish and store their IDs in a set
#         valid_fish = []
#         valid_ids = set()
#
#         for fish in json_data["fish"]:
#             if fish["length"] > min_length and fish["travel_dist"] > min_travel:
#                 valid_fish.append(fish)
#                 valid_ids.add(fish["id"])
#
#         json_data["fish"] = valid_fish
#
#         for frame in json_data["frames"]:
#             frame["fish"] = [
#                 fish for fish in frame["fish"] if fish["fish_id"] in valid_ids
#             ]
#
#         if output_path:
#             with open(output_path, "w") as output:
#                 json.dump(json_data, output, indent=2)
#
#         return json_data
#
#     def _process_track(self, track, detections):
#         conf = self._match_confidence(track, detections)
#         fish_id = int(track[4])
#         self.fish_ids[fish_id] += 1
#         return {
#             "fish_id": fish_id,
#             "bbox": list(track[:4]),
#             "visible": 1,
#             "human_labeled": 0,
#             "conf": conf,
#         }
#
#     def _match_confidence(self, track, detections):
#         min_score, conf = float("inf"), 0
#         if isinstance(self.algorithm, Sort):
#             for det in detections:
#                 score = np.sum(np.abs(det[:4] - track[:4]))
#                 if score < min_score:
#                     min_score, conf = score, det[4]
#
#         elif isinstance(self.algorithm, ByteTracker):
#             for det_set in detections:
#                 for det in det_set:
#                     score = np.sum(np.abs(det[:4] - track[:4]))
#                     if score < min_score:
#                         min_score, conf = score, det[4]
#
#         return conf
#
#     def _map_fish_ids(self, json_data):
#         """Map valid fish IDs to 0, 1, 2, ..."""
#         fish_id_map = {}
#         for fish_id, count in self.fish_ids.items():
#             if count >= self.min_hits:
#                 fish_id_map[fish_id] = len(fish_id_map)
#
#         fish_id_map = {
#             fish_id: i
#             for i, (fish_id, count) in enumerate(self.fish_ids.items())
#             if count >= self.min_hits
#         }
#
#         tracks = {v: [] for v in fish_id_map.values()}
#         for frame in json_data["frames"]:
#             for bbox in frame["fish"]:
#                 if bbox["fish_id"] in fish_id_map:
#                     track_id = fish_id_map[bbox["fish_id"]]
#                     tracks[track_id].append((bbox["bbox"], frame["frame_num"]))
#
#         return fish_id_map, tracks
#
#
def run_tracker(
    low_preds,
    high_preds,
    image_meter_width,
    image_meter_height,
    tracking_config,
    reverse=False,
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

    tracker = Tracker(clip_info, tracking_config)

    with tqdm(
        total=len(low_preds), desc="Running tracking", ncols=0, disable=not verbose
    ) as pbar:
        for i, key in enumerate(sorted(low_preds.keys(), reverse=reverse)):
            if gp:
                gp(i / len(low_preds), pbar.__str__())
            low_boxes, high_boxes = low_preds[key], high_preds[key]
            boxes = (
                (low_boxes, high_boxes)
                if low_boxes is not None and high_boxes is not None
                else (np.empty((0, 5)), np.empty((0, 5)))
            )
            tracker.run(boxes)
            pbar.update(1)

    json_data = tracker.finalize(
        min_length=tracking_config.min_length, min_travel=tracking_config.min_travel
    )

    return json_data
