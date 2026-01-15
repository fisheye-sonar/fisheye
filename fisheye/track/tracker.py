import json
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
import structlog
from tqdm import tqdm

from fisheye.configs.inference import TrackerConfig, FishSizeConfig, TrackerOutput
from fisheye.enums import TrackingMethod
from fisheye.track.bytetrack import ByteTracker
from fisheye.track.sort import Sort
from fisheye.track.utils import FishMetrics

logger = structlog.get_logger()

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
                "min_hits": 0,  # TODO (MVH) - why aren't we being consistent with min_hits?
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
            raise ValueError(f"Tracking method `{algorithm}` is not supported.")

        return tracker_cls(**args)

    def update(self, dets=np.empty((0, 5))):
        """Updates the tracker with new detections. Boxes should be given in normalized [x1,y1,x2,y2,c]"""
        new_frame_entries = []
        for track in self.algorithm.update(dets):

            # Match confidence with correct track
            conf = 0
            min_score = 1000000
            best_bbox_index = None  # which low/high bbox did it match to (if that detector returned multiple bboxes)
            det_index = None  # did it match to the low [0] or high [1] detection?
            if TrackingMethod.SORT == self.algorithm.type:
                conf, min_score, bbox_index, improved_match = (
                    self.algorithm.match_confidence_to_track(
                        track, dets, conf, min_score
                    )
                )

            elif TrackingMethod.BYTETRACK == self.algorithm.type:
                # dets[0] = low conf, dets[1] = high conf
                for i, det_group in enumerate(dets):
                    conf, min_score, bbox_index, improved_match = (
                        self.algorithm.match_confidence_to_track(
                            track, det_group, conf, min_score
                        )
                    )

                    if improved_match:
                        det_index = i
                        best_bbox_index = bbox_index

            # Assign Track
            self.fish_ids[int(track[4])] += 1
            new_frame_entries.append(
                {
                    "fish_id": int(track[4]),
                    "bbox": list(track[:4]),
                    "conf": conf,
                    "bbox_index": best_bbox_index,
                    "det_index": det_index,
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
    ):
        # Determine valid fish IDs based on min_hits threshold
        valid_fish_ids = {
            fish_id
            for fish_id, count in self.fish_ids.items()
            if count >= self.min_hits
        }

        # Early return if no valid tracks
        if not valid_fish_ids:
            result = {
                **self.json_data,
                "fish": [],
                "frames": [
                    {"frame_num": frame["frame_num"], "fish": []}
                    for frame in self.json_data["frames"]
                ],
            }
            if output_path is not None:
                with open(output_path, "w") as output:
                    json.dump(result, output, indent=2)
            return result

        tracks = defaultdict(list)

        for frame in self.json_data["frames"]:
            for bbox_data in frame["fish"]:
                fish_id = bbox_data["fish_id"]
                if fish_id in valid_fish_ids:
                    tracks[fish_id].append(
                        {
                            "bbox": bbox_data["bbox"],
                            "frame_num": frame["frame_num"],
                            "bbox_index": bbox_data["bbox_index"],
                            "det_index": bbox_data["det_index"],
                        }
                    )

        # Create ID mapping (valid fish IDs -> 0, 1, 2, ...)
        fish_id_map = {
            old_id: new_id for new_id, old_id in enumerate(sorted(valid_fish_ids))
        }

        # Build fish summary with metrics
        fish_summary = []
        for old_id, boxes in tracks.items():
            track_id = fish_id_map[old_id]
            start_bbox = boxes[0]["bbox"]
            end_bbox = boxes[-1]["bbox"]

            fish_entry = {
                "id": track_id,
                "length": -1,  # Will be populated by add_lengths
                "travel_dist": FishMetrics.get_travel_distance(
                    start_bbox,
                    end_bbox,
                    self.json_data["image_meter_width"],
                    self.json_data["image_meter_height"],
                ),
                "start_frame_index": boxes[0]["frame_num"],
                "end_frame_index": boxes[-1]["frame_num"],
            }
            fish_summary.append(fish_entry)

        # Sort by track ID for consistency
        fish_summary.sort(key=lambda x: x["id"])

        # Add length estimates
        result = {
            **self.json_data,
            "fish": fish_summary,
        }
        result = FishMetrics.add_lengths(result)

        # Determine which IDs to keep after length/travel filtering
        if min_length != -1.0:
            valid_track_ids = {
                fish["id"]
                for fish in result["fish"]
                # if fish["length"] > min_length and fish["travel_dist"] > min_travel
                if fish["travel_dist"] > min_travel
            }
            result["fish"] = [
                fish for fish in result["fish"] if fish["id"] in valid_track_ids
            ]
        else:
            valid_track_ids = {fish["id"] for fish in result["fish"]}

        # PASS 2: Remap and filter frames in a single pass
        # This combines the old remapping pass and filtering pass
        filtered_frames = []
        for frame in self.json_data["frames"]:
            filtered_fish = []
            for bbox_data in frame["fish"]:
                old_id = bbox_data["fish_id"]
                if old_id in fish_id_map:
                    new_id = fish_id_map[old_id]
                    if new_id in valid_track_ids:
                        filtered_fish.append(
                            {
                                **bbox_data,
                                "fish_id": new_id,
                            }
                        )

            # Keep fish sorted by ID
            filtered_fish.sort(key=lambda x: x["fish_id"])
            filtered_frames.append(
                {
                    "frame_num": frame["frame_num"],
                    "fish": filtered_fish,
                }
            )

        result["frames"] = filtered_frames

        if output_path:
            with open(output_path, "w") as output:
                json.dump(result, output, indent=2)

        return result


def run_tracker(
    low_preds,
    high_preds,
    image_meter_width,
    image_meter_height,
    tracking_config,
    min_length=FishSizeConfig.min_length,
    gp=None,
    verbose=False,
):
    """Factory method to run tracker."""
    logger.info(
        "initialized_tracker",
        tracker_type=tracking_config.type,
        max_age=tracking_config.max_age,
        min_hits=tracking_config.min_hits,
        iou_threshold=tracking_config.iou_threshold,
        reverse=tracking_config.reverse,
    )
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

            # MAH 2025-11-26 12:41:56 TODO do we want to exclude when there is no high pred?,  i think it should be
            boxes = (
                low_boxes if low_boxes is not None else np.empty((0, 5)),
                high_boxes if high_boxes is not None else np.empty((0, 5)),
            )

            tracker.update(boxes)
            pbar.update(1)

    json_data = tracker.finalize(
        min_length=min_length, min_travel=tracking_config.min_travel
    )

    output = TrackerOutput.dict_to_dataclass(json_data)

    return output
