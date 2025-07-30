import structlog
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd

from fisheye.count.base import BaseCounter
from fisheye.enums import CountingMethod

logger = structlog.getLogger(__name__)


class LOICounter(BaseCounter):
    type = CountingMethod.LOI
    """Count fish by crossing the Line of Interest (LOI)."""

    def count(self, df: pd.DataFrame, line: float = 0.5):
        """Count fish that cross the LOI. Defaults to using center line."""
        grouped_tracks = df.sort_values(by=["frame"], ascending=True).groupby("id")
        track_counts = defaultdict(lambda: {"left": 0, "right": 0})
        crossing_frames = {"left": [], "right": []}

        for track, track_df in grouped_tracks:
            x_coords = track_df["kp_x"].values
            frames = track_df["frame"].values
            first_x = x_coords[0]
            last_x = x_coords[-1]

            # Find the index of the point closest to the line
            distances = np.abs(x_coords - line)
            closest_idx = np.argmin(distances)

            x_center = track_df["x_center"].values[closest_idx]
            y_center = track_df["y_center"].values[closest_idx]
            width = track_df["width"].values[closest_idx]
            height = track_df["height"].values[closest_idx]
            closest_frame = (
                int(frames[closest_idx]) - 1
            )  # Subtract 1 since it was for MOT format
            bbox = [x_center, y_center, width, height]

            # Determine initial and final positions
            if last_x <= line < first_x:
                track_counts[track]["left"] += 1
                crossing_frames["left"].append((track, closest_frame, bbox))

            elif last_x >= line > first_x:
                track_counts[track]["right"] += 1
                crossing_frames["right"].append((track, closest_frame, bbox))

        return self._calculate_absolute_counts(track_counts), crossing_frames

    @staticmethod
    def _calculate_absolute_counts(track_counts):
        """Calculate absolute counts for left and right."""
        absolute_left_counts = 0
        absolute_right_counts = 0

        for track, count in track_counts.items():
            net_counts = count["left"] - count["right"]
            if net_counts > 0:
                absolute_left_counts += 1
            else:
                absolute_right_counts += 1

        return absolute_left_counts, absolute_right_counts


class Count:
    """Main class to handle different counting methods."""

    def __init__(self, protocol: str = "loi"):
        """Initialize Count with a specific counting protocol."""
        self.protocol = protocol

        if self.protocol == CountingMethod.LOI:
            self.counter = LOICounter()
        else:
            raise ValueError(f"Protocol '{self.protocol}' is not supported.")

    def count(self, tracks: List[dict]):
        """Count fish using the selected protocol.

        Args:
            tracks (dict): Bounding boxes in YOLO format with the following keys: 'frame', 'id', 'x_center', 'y_center',
            'width', 'height', 'conf'.

        Returns:
            tuple: (absolute_left_count, absolute_right_count), crossing frames.
        """
        logger.info(f"initialized_counter", type=self.protocol)
        df = pd.DataFrame(tracks)
        if not df.empty:
            # Calculate the center point of the bounding box
            df["kp_x"] = df["x_center"] + df["width"] / 2
            df["kp_y"] = df["y_center"] + df["height"] / 2

            return self.counter.count(df)

        logger.warning(f"No tracks present.")
        # If no tracks present (empty dataframe) return 0 for both left and right counts
        return (0, 0), None
