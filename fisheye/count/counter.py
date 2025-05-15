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

    def count(self, mot_df: pd.DataFrame, line: float = 0.5):
        """Count fish that cross the LOI. Defaults to using center line."""
        grouped_tracks = mot_df.sort_values(by=["frame"], ascending=True).groupby("id")
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

            bb_left = track_df["bb_left"].values[closest_idx]
            bb_top = track_df["bb_top"].values[closest_idx]
            bb_width = track_df["bb_width"].values[closest_idx]
            bb_height = track_df["bb_height"].values[closest_idx]
            closest_frame = (
                int(frames[closest_idx]) - 1
            )  # Subtract 1 since it was for MOT format
            bbox = [bb_left, bb_top, bb_width, bb_height]

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
            tracks (dict): Tracks in MOT format with the following keys: 'frame', 'id', 'bb_left', 'bb_top',
            'bb_height', 'conf'.

        Returns:
            tuple: (absolute_left_count, absolute_right_count)
        """
        logger.info(f"initialized_counter", type=self.protocol)
        mot_df = pd.DataFrame(tracks)
        if not mot_df.empty:
            # Calculate the center point of the bounding box
            mot_df["kp_x"] = mot_df["bb_left"] + mot_df["bb_width"] / 2
            mot_df["kp_y"] = mot_df["bb_top"] + mot_df["bb_height"] / 2

            return self.counter.count(mot_df)

        logger.warning(f"No tracks present.")
        # If no tracks present (empty dataframe) return 0 for both left and right counts
        return (0, 0), None
