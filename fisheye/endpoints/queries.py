"""Translate retained tracker observations to explicit pixel-space queries."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .types import EndpointQuery


@dataclass(frozen=True)
class DetectionRecord:
    detection_id: str
    frame_id: int
    source_group: int
    source_index: int
    bbox_xyxy_px: tuple
    detector_score: float


def detection_records(low_preds, high_preds, frame_ids, width, height):
    records = {}
    scale = np.array([width, height, width, height], dtype=float)
    if set(low_preds) != set(high_preds) or set(low_preds) != set(frame_ids):
        raise ValueError("Frame IDs and low/high detection keys must match exactly")
    for key in frame_ids:
        frame_id = frame_ids[key]
        for group, predictions in enumerate((low_preds, high_preds)):
            boxes = predictions[key]
            if boxes is None:
                continue
            for index, row in enumerate(boxes):
                values = np.asarray(row, dtype=float)
                if len(values) < 5 or not np.isfinite(values[:5]).all():
                    raise ValueError("Nonfinite or malformed detector output")
                box = tuple((values[:4] * scale).tolist())
                records[(frame_id, group, index)] = DetectionRecord(
                    f"{frame_id}:{group}:{index}",
                    frame_id,
                    group,
                    index,
                    box,
                    float(values[4]),
                )
    return records


def build_queries(tracker_output, records, clip_id, selector):
    """Skip unassociated observations explicitly; never guess a nearby fish."""
    tracks, skipped = defaultdict(list), []
    for frame in tracker_output.frames:
        for fish in frame.fish:
            record = records.get((frame.frame_num, fish.det_index, fish.bbox_index))
            if record is None:
                skipped.append(
                    {
                        "frame_id": frame.frame_num,
                        "track_id": fish.id,
                        "reason": "missing_detection_association",
                    }
                )
                continue
            tracks[fish.id].append(record)
    queries = []
    for track_id in sorted(tracks):
        observations = sorted(tracks[track_id], key=lambda r: r.frame_id)
        frame_ids = tuple(record.frame_id for record in observations)
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError(f"Duplicate observations in track {track_id}")
        for position, record in enumerate(observations):
            context = selector.resolve(frame_ids, position)
            queries.append(
                EndpointQuery(
                    query_id=f"{clip_id}:{track_id}:{record.frame_id}",
                    clip_id=clip_id,
                    frame_id=record.frame_id,
                    track_id=track_id,
                    detection_id=record.detection_id,
                    bbox_xyxy_px=record.bbox_xyxy_px,
                    context_frame_ids=tuple(context.frame_ids),
                    context_repeated=tuple(context.repeated),
                )
            )
    return queries, skipped
