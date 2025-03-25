from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd

from fisheye.count.base import BaseCounter


class LOICounter(BaseCounter):
    """Count fish by crossing the Line of Interest (LOI)."""

    def count(self, mot_df: pd.DataFrame, line: float = 0.5):
        """Count fish that cross the center line."""
        grouped_tracks = mot_df.sort_values(by=["frame"], ascending=True).groupby("id")
        track_counts = defaultdict(lambda: {"left": 0, "right": 0})

        for track, track_df in grouped_tracks:
            x_coords = track_df["kp_x"].values
            first_x = x_coords[0]
            last_x = x_coords[-1]

            # Determine initial and final positions
            if last_x <= line < first_x:
                track_counts[track]["left"] += 1
            elif last_x >= line > first_x:
                track_counts[track]["right"] += 1

            # Detect line crossings
            crossings = np.where((x_coords[:-1] - line) * (x_coords[1:] - line) <= 0)[0]

            if len(crossings) > 0:
                for i in crossings:
                    x1, x2 = x_coords[i], x_coords[i + 1]

                    if x1 == line and x2 == line:
                        continue

                    if x1 < x2 or (x1 == line and x2 > line):
                        track_counts[track]["right"] += 1

                    elif x1 > x2 or (x1 == line and x2 < line):
                        track_counts[track]["left"] += 1

        return self._calculate_counts(track_counts)

    # def _calculate_counts(self, track_counts):
    #     """Calculate net counts for left and right."""
    #     print(track_counts)
    #     crossings_df = pd.DataFrame.from_dict(track_counts, orient="index")
    #     absolute_left_counts = int((crossings_df["left"] > crossings_df["right"]).sum())
    #     absolute_right_counts = int((crossings_df["left"] <= crossings_df["right"]).sum())
    #
    #     return absolute_left_counts, absolute_right_counts

    def _calculate_counts(self, track_counts):
        """Calculate net counts for left and right."""
        absolute_left_counts = 0
        absolute_right_counts = 0

        for track, count in track_counts.items():
            net_counts = count["left"] - count["right"]
            if net_counts > 0:
                absolute_left_counts += 1
            else:
                absolute_right_counts += 1

        return absolute_left_counts, absolute_right_counts


class CounterFactory:
    """Factory to create appropriate counter based on protocol."""

    @staticmethod
    def get_counter(protocol: str = "LOI"):
        """Return counter class based on protocol."""
        if protocol == "LOI":
            return LOICounter()
        else:
            raise ValueError(f"Protocol '{protocol}' is not supported.")


class Count:
    """Main class to handle different counting methods."""

    def __init__(self, protocol: str = "LOI"):
        """Initialize Count with a specific counting protocol."""
        self.protocol = protocol
        self.counter = CounterFactory.get_counter(protocol)

    def count(self, tracks: List[dict]):
        """Count fish using the selected protocol.

        Args:
            tracks (dict): MOT tracks data.

        Returns:
            tuple: (left_count, right_count)
        """
        mot_df = pd.DataFrame(tracks)
        mot_df["kp_x"] = mot_df["bb_left"] + mot_df["bb_width"] / 2
        mot_df["kp_y"] = mot_df["bb_top"] + mot_df["bb_height"] / 2

        return self.counter.count(mot_df)
