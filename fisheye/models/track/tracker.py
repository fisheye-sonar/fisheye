import json
from collections import Counter
from copy import deepcopy

import numpy as np

from .utils import FishMetrics
from .bytetrack import Associate
from .sort import Sort


class Tracker:
    def __init__(self, clip_info, config):
        self.algorithm = config.type(**config)
        self.fish_ids = Counter()
        self.reverse = config.reverse
        self.min_hits = config.min_hits
        self.json_data = deepcopy(clip_info)
        self.frame_id = (
            self.json_data["end_frame"]
            if self.reverse
            else self.json_data["start_frame"]
        )
        self.json_data["frames"] = []

    def run(self, detections=np.empty((0, 5))):
        """Boxes should be given in normalized [x1,y1,x2,y2,c]"""
        new_frame_entries = [
            self._process_track(track, detections)
            for track in self.algorithm.update(detections)
        ]
        self.json_data["frames"].append(
            {
                "frame_num": self.frame_id,
                "fish": sorted(new_frame_entries, key=lambda k: k["fish_id"]),
            }
        )
        self.frame_id += -1 if self.reverse else 1

    def finalize(self, output_path=None, min_length=-1.0, min_travel=-1.0):
        """Process tracking results with valid fish IDs."""
        json_data = deepcopy(self.json_data)
        fish_id_map, tracks = self._map_fish_ids(json_data)

        for frame in json_data["frames"]:
            frame["fish"] = sorted(
                [
                    {**entry, "fish_id": fish_id_map[entry["fish_id"]]}
                    for entry in frame["fish"]
                    if entry["fish_id"] in fish_id_map
                ],
                key=lambda k: k["fish_id"],
            )

        # create summary 'fish' entry for json data
        json_data["fish"] = []
        for track_id, boxes in tracks.items():
            fish_entry = {}
            fish_entry["id"] = track_id
            # TODO (MVH) - why is this hard coded?
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

        json_data = FishMetrics.add_lengths(json_data)

        # Filter valid fish and store their IDs in a set
        valid_fish = []
        valid_ids = set()

        for fish in json_data["fish"]:
            if fish["length"] > min_length and fish["travel_dist"] > min_travel:
                valid_fish.append(fish)
                valid_ids.add(fish["id"])

        json_data["fish"] = valid_fish

        for frame in json_data["frames"]:
            frame["fish"] = [
                fish for fish in frame["fish"] if fish["fish_id"] in valid_ids
            ]

        if output_path:
            with open(output_path, "w") as output:
                json.dump(json_data, output, indent=2)

        return json_data

    def _process_track(self, track, detections):
        conf = self._match_confidence(track, detections)
        fish_id = int(track[4])
        self.fish_ids[fish_id] += 1
        return {
            "fish_id": fish_id,
            "bbox": list(track[:4]),
            "visible": 1,
            "human_labeled": 0,
            "conf": conf,
        }

    def _match_confidence(self, track, detections):
        min_score, conf = float("inf"), 0
        if isinstance(self.algorithm, Sort):
            for det in detections:
                score = np.sum(np.abs(det[:4] - track[:4]))
                if score < min_score:
                    min_score, conf = score, det[4]
        elif isinstance(self.algorithm, Associate):
            for det_set in detections:
                for det in det_set:
                    score = np.sum(np.abs(det[:4] - track[:4]))
                    if score < min_score:
                        min_score, conf = score, det[4]
        return conf

    def _map_fish_ids(self, json_data):
        """Map valid fish IDs to 0, 1, 2, ..."""
        fish_id_map = {}
        for fish_id, count in self.fish_ids.items():
            if count >= self.min_hits:
                fish_id_map[fish_id] = len(fish_id_map)

        fish_id_map = {
            fish_id: i
            for i, (fish_id, count) in enumerate(self.fish_ids.items())
            if count >= self.min_hits
        }
        tracks = {v: [] for v in fish_id_map.values()}
        for frame in json_data["frames"]:
            for bbox in frame["fish"]:
                if bbox["fish_id"] in fish_id_map:
                    track_id = fish_id_map[bbox["fish_id"]]
                    tracks[track_id].append((bbox["bbox"], frame["frame_num"]))
        return fish_id_map, tracks
